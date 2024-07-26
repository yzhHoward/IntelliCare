import os
import math
import numpy as np
import pandas as pd
import pickle
import shutil
from itertools import chain
from scipy.sparse import csr_matrix, vstack
import torch
import threading
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm

from utils.mimic import MIMIC3Dataset, MIMIC4Dataset
from utils.tasks import set_task, mimic3_fn_for_generation, mimic4_fn_for_generation, mortality_prediction_mimic3_fn, mortality_prediction_mimic4_fn
from utils.cluster import perform_clustering

os.environ["TOKENIZERS_PARALLELISM"] = "false"


model_id = "../Meta-Llama-3-8B-Instruct"
# model_id = "../Mistral-7B-Instruct-v0.3"
dataset = 'mimiciii'
gpus=4
batch_size = 32
seed = 2 if dataset == 'mimiciii' else 1
n = 8
n_components = 5000
use_cluster = True
export_dir = f'./export/{dataset}/llama-cluster-{n}x'
if not os.path.exists(export_dir):
    os.makedirs(export_dir)


def file_processing(file_name, content):
    with open(file_name, 'w') as f:
        f.write(content)


def clustering():
    if not os.path.exists(f"export/{dataset}/reduced_embeddings.pkl"):
        multihot_index = {}
        for table, mapping in base_dataset.name_mapping.items():
            multihot_index[table] = {}
            for i, key in enumerate(mapping.keys()):
                multihot_index[table][key] = i

        multihot_tables = []
        name_dict = {
            "conditions": "DIAGNOSES_ICD" if dataset == "mimiciii" else "diagnoses_icd",
            "procedures": "PROCEDURES_ICD" if dataset == "mimiciii" else "procedures_icd"
        }
        for patient in tqdm(sample_dataset.samples, desc="Generating multihot matrix"):
            multihot_table = []
            for i, table in enumerate(name_dict.keys()):
                multihot_table.append(np.zeros((len(multihot_index[name_dict[table]]),), dtype=np.bool_))
                for event in patient[table][0]:
                    multihot_table[i][multihot_index[name_dict[table]][event]] = True
            multihot_tables.append(csr_matrix(np.concatenate(multihot_table)))
        multihot_tables = vstack(multihot_tables)
    else:
        multihot_tables = None

    cluster_reference_probs, cluster_prob, cluster_label = perform_clustering(dataset, multihot_tables, train_index=train_index, n_components=n_components)
    
    readmission = torch.tensor([patient["readmission_label"] for patient in sample_dataset.samples])[train_index].to(torch.float64)
    mortality = torch.tensor([patient["mortality_label"] for patient in sample_dataset.samples])[train_index].to(torch.float64)
    los = torch.tensor([patient["los_days"] for patient in sample_dataset.samples])[train_index].to(torch.float64)
    for i, patient in enumerate(sample_dataset):
        patient["cluster_prob"] = cluster_prob[i]
        patient["cluster_label"] = cluster_label[i]
    age = (cluster_reference_probs.t() @ torch.tensor([patient["age"] for patient in sample_dataset.samples])[train_index].to(torch.float64)) / cluster_reference_probs.sum(0)
    gender = (cluster_reference_probs.t() @ torch.tensor([patient["gender"] == 'M' for patient in sample_dataset.samples])[train_index].to(torch.float64)) / cluster_reference_probs.sum(0)
    diagnose_num = (cluster_reference_probs.t() @ torch.tensor([len(patient["condition_names"][0]) for patient in sample_dataset.samples])[train_index].to(torch.float64)) / cluster_reference_probs.sum(0)
    procedure_num = (cluster_reference_probs.t() @ torch.tensor([len(patient["procedure_names"][0]) for patient in sample_dataset.samples])[train_index].to(torch.float64)) / cluster_reference_probs.sum(0)
    drug_num = (cluster_reference_probs.t() @ torch.tensor([len(patient["drug_names"][0]) for patient in sample_dataset.samples])[train_index].to(torch.float64)) / cluster_reference_probs.sum(0)
    cluster_statistic = {
        "patient_num": cluster_reference_probs.sum(0).tolist(),
        "age": age,
        "gender": gender,
        "diagnose_num": diagnose_num,
        "procedure_num": procedure_num,
        "drug_num": drug_num,
        "readmission": (cluster_reference_probs.t() @ readmission) / cluster_reference_probs.sum(0),
        "mortality": (cluster_reference_probs.t() @ mortality) / cluster_reference_probs.sum(0),
        "los": (cluster_reference_probs.t() @ los) / cluster_reference_probs.sum(0),
    }
    return cluster_statistic


def generate_prompt(cluster=False):
    if cluster:
        cluster_statistic = clustering()
    if 'Mistral' in model_id:
        data = {'text': [], 'name': []}
        for patient in sample_dataset:
            assert patient['gender'] == 'M' or patient['gender'] == 'F'
            text = "[INST] "
            if isinstance(patient['diagnosis'], str):
                text += f"There is a {patient['age']}-year-old {'male' if patient['gender'] == 'M' else 'female'} patient who is admitted to the ICU since {patient['diagnosis'].lower()}.\n"
            else:
                text += f"There is a {patient['age']}-year-old {'male' if patient['gender'] == 'M' else 'female'} patient who is admitted to the ICU.\n"
            text += f"\nThis patient is diagnosed with {len(patient['condition_names'][0])} conditions:\n"
            for i, name in enumerate(patient['condition_names'][0], 1):
                text += f"- {name}\n"
            text += f"\nThis patient underwent {len(patient['procedure_names'][0])} procedures:\n"
            for i, name in enumerate(patient['procedure_names'][0], 1):
                text += f"- {name}\n"
            text += f"\nThis patient used {len(patient['drug_names'][0])} medications:\n"
            for i, name in enumerate(patient['drug_names'][0], 1):
                text += f"- {name}\n"
            if cluster:
                if len(patient['cluster_label']) == 1:
                    text += f"\nThe statistics of the cohort that the patient may belong to (for reference only) is:"
                    text += f"\n{cluster_statistic['patient_num'][patient['cluster_label'][0]]:.0f} in total. The average age is {cluster_statistic['age'][patient['cluster_label'][0]]:.0f} years old, {cluster_statistic['gender'][patient['cluster_label'][0]]:.0%} of the patients are male, and the average number of conditions, procedures, and medications are {cluster_statistic['diagnose_num'][patient['cluster_label'][0]]:.0f}, {cluster_statistic['procedure_num'][patient['cluster_label'][0]]:.0f}, and {cluster_statistic['drug_num'][patient['cluster_label'][0]]:.0f}, respectively. The average length of stay is {cluster_statistic['los'][patient['cluster_label'][0]]:.0f} days. The readmission rate in 30 days is {cluster_statistic['readmission'][patient['cluster_label'][0]]:.0%}. The mortality rate at the next admission is {cluster_statistic['mortality'][patient['cluster_label'][0]]:.0%}.\n"
                else:
                    text += f"\nThe statistics of the cohorts that the patient may belong to (for reference only) are:\n"
                    for i, label in enumerate(patient['cluster_label']):
                        text += f"- Cohort {i + 1} ({patient['cluster_prob'][i]:.0%} probability): {cluster_statistic['patient_num'][label]:.0f} in total. The average age is {cluster_statistic['age'][label]:.0f} years old, {cluster_statistic['gender'][label]:.0%} of the patients are male, and the average number of conditions, procedures, and medications are {cluster_statistic['diagnose_num'][label]:.0f}, {cluster_statistic['procedure_num'][label]:.0f}, and {cluster_statistic['drug_num'][label]:.0f}, respectively. The average length of stay is {cluster_statistic['los'][label]:.0f} days. The readmission rate in 30 days is {cluster_statistic['readmission'][label]:.0%}. The mortality rate at the next admission is {cluster_statistic['mortality'][label]:.0%}.\n"
            text += "\nBased on the patient information provided above, please briefly summarize the patient's condition. Assess risks with brief explanations for length of stay, 30-day readmission, and outcome of next admission. Your analysis should reflect the unique aspects of this patient's condition and treatment."
            text += " [/INST]"
            data['text'].append(text)
            data['name'].append(patient['visit_id'])
    elif 'Llama-3' in model_id:
        data = {'input_ids': [], 'name': []}
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        for patient in sample_dataset:
            assert patient['gender'] == 'M' or patient['gender'] == 'F'
            text = ''
            if isinstance(patient['diagnosis'], str):
                text += f"There is a {patient['age']}-year-old {'male' if patient['gender'] == 'M' else 'female'} patient who is admitted to the ICU since {patient['diagnosis'].lower()}.\n"
            else:
                text += f"There is a {patient['age']}-year-old {'male' if patient['gender'] == 'M' else 'female'} patient who is admitted to the ICU.\n"
            text += f"\nThis patient is diagnosed with {len(patient['condition_names'][0])} conditions:\n"
            for i, name in enumerate(patient['condition_names'][0], 1):
                text += f"- {name}\n"
            text += f"\nThis patient underwent {len(patient['procedure_names'][0])} procedures:\n"
            for i, name in enumerate(patient['procedure_names'][0], 1):
                text += f"- {name}\n"
            text += f"\nThis patient used {len(patient['drug_names'][0])} medications:\n"
            for i, name in enumerate(patient['drug_names'][0], 1):
                text += f"- {name}\n"
            if cluster:
                if len(patient['cluster_label']) == 1:
                    text += f"\nThe statistics of the cohort that the patient may belong to (for reference only) is:"
                    text += f"\n{cluster_statistic['patient_num'][patient['cluster_label'][0]]:.0f} in total. The average age is {cluster_statistic['age'][patient['cluster_label'][0]]:.0f} years old, {cluster_statistic['gender'][patient['cluster_label'][0]]:.0%} of the patients are male, and the average number of conditions, procedures, and medications are {cluster_statistic['diagnose_num'][patient['cluster_label'][0]]:.0f}, {cluster_statistic['procedure_num'][patient['cluster_label'][0]]:.0f}, and {cluster_statistic['drug_num'][patient['cluster_label'][0]]:.0f}, respectively. The average length of stay is {cluster_statistic['los'][patient['cluster_label'][0]]:.0f} days. The readmission rate in 30 days is {cluster_statistic['readmission'][patient['cluster_label'][0]]:.0%}. The mortality rate at the next admission is {cluster_statistic['mortality'][patient['cluster_label'][0]]:.0%}.\n"
                else:
                    text += f"\nThe statistics of the cohorts that the patient may belong to (for reference only) are:\n"
                    for i, label in enumerate(patient['cluster_label']):
                        text += f"- Cohort {i + 1} ({patient['cluster_prob'][i]:.0%} probability): {cluster_statistic['patient_num'][label]:.0f} in total. The average age is {cluster_statistic['age'][label]:.0f} years old, {cluster_statistic['gender'][label]:.0%} of the patients are male, and the average number of conditions, procedures, and medications are {cluster_statistic['diagnose_num'][label]:.0f}, {cluster_statistic['procedure_num'][label]:.0f}, and {cluster_statistic['drug_num'][label]:.0f}, respectively. The average length of stay is {cluster_statistic['los'][label]:.0f} days. The readmission rate in 30 days is {cluster_statistic['readmission'][label]:.0%}. The mortality rate at the next admission is {cluster_statistic['mortality'][label]:.0%}.\n"
            text += "\nBased on the patient information provided above, please briefly summarize the patient's condition. Assess risks with brief explanations for length of stay, 30-day readmission, and outcome of next admission. Your analysis should reflect the unique aspects of this patient's condition and treatment."
            input_ids = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": text}
                ],
                add_generation_prompt=True,
                return_tensors="pt"
            ).tolist()
            if len(input_ids[0]) > 8192:
                print(f"Warning: The prompt length of patient {patient['visit_id']} is {len(input_ids[0])}, which exceeds the maximum length of {8192}. Trying to regenerate prompt without medications.")
                text = ''
                if isinstance(patient['diagnosis'], str):
                    text += f"There is a {patient['age']}-year-old {'male' if patient['gender'] == 'M' else 'female'} patient who is admitted to the ICU since {patient['diagnosis'].lower()}.\n"
                else:
                    text += f"There is a {patient['age']}-year-old {'male' if patient['gender'] == 'M' else 'female'} patient who is admitted to the ICU.\n"
                text += "This patient is diagnosed with:\n"
                text += f"\nThis patient is diagnosed with {len(patient['condition_names'][0])} conditions:\n"
                for i, name in enumerate(patient['condition_names'][0], 1):
                    text += f"- {name}\n"
                text += f"\nThis patient underwent {len(patient['procedure_names'][0])} procedures:\n"
                for i, name in enumerate(patient['procedure_names'][0], 1):
                    text += f"- {name}\n"
                if cluster:
                    if len(patient['cluster_label']) == 1:
                        text += f"\nThe statistics of the cohort that the patient may belong to (for reference only) is:"
                        text += f"\n{cluster_statistic['patient_num'][patient['cluster_label'][0]]:.0f} in total. The average age is {cluster_statistic['age'][patient['cluster_label'][0]]:.0f} years old, {cluster_statistic['gender'][patient['cluster_label'][0]]:.0%} of the patients are male, and the average number of conditions, procedures, and medications are {cluster_statistic['diagnose_num'][patient['cluster_label'][0]]:.0f}, {cluster_statistic['procedure_num'][patient['cluster_label'][0]]:.0f}, and {cluster_statistic['drug_num'][patient['cluster_label'][0]]:.0f}, respectively. The average length of stay is {cluster_statistic['los'][patient['cluster_label'][0]]:.0f} days. The readmission rate in 30 days is {cluster_statistic['readmission'][patient['cluster_label'][0]]:.0%}. The mortality rate at the next admission is {cluster_statistic['mortality'][patient['cluster_label'][0]]:.0%}.\n"
                    else:
                        text += f"\nThe statistics of the cohorts that the patient may belong to (for reference only) are:\n"
                        for i, label in enumerate(patient['cluster_label']):
                            text += f"- Cohort {i + 1} ({patient['cluster_prob'][i]:.0%} probability): {cluster_statistic['patient_num'][label]:.0f} in total. The average age is {cluster_statistic['age'][label]:.0f} years old, {cluster_statistic['gender'][label]:.0%} of the patients are male, and the average number of conditions, procedures, and medications are {cluster_statistic['diagnose_num'][label]:.0f}, {cluster_statistic['procedure_num'][label]:.0f}, and {cluster_statistic['drug_num'][label]:.0f}, respectively. The average length of stay is {cluster_statistic['los'][label]:.0f} days. The readmission rate in 30 days is {cluster_statistic['readmission'][label]:.0%}. The mortality rate at the next admission is {cluster_statistic['mortality'][label]:.0%}.\n"
                text += "\nBased on the patient information provided above, please briefly summarize the patient's condition. Assess risks with brief explanations for length of stay, 30-day readmission, and outcome of next admission. Your analysis should reflect the unique aspects of this patient's condition and treatment."
                input_ids = tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": text}
                    ],
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).tolist()
                assert len(input_ids[0]) < 8192
            data['input_ids'].append(input_ids[0])
            data['name'].append(patient['visit_id'])
    return data


if dataset == "mimiciii":
    base_dataset = MIMIC3Dataset(
        root="data/mimiciii/",
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        dev=False,
        refresh_cache=False,
    )
    base_dataset.load_name_mapping({"DIAGNOSES_ICD": "ICD9CM", "PROCEDURES_ICD": "ICD9PROC"})
    if not os.path.exists(f"export/{dataset}/mortality.pkl"):
        target_dataset = MIMIC3Dataset(
            root="data/mimiciii/",
            tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
            code_mapping={"ICD9CM": "CCSCM", "ICD9PROC": "CCSPROC", "NDC": ("ATC", {"target_kwargs": {"level": 3}})},
            dev=False,
            refresh_cache=False,
        )
    else:
        target_dataset = None
else:
    base_dataset = MIMIC4Dataset(
        root="data/mimiciv/",
        tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        dev=False,
        refresh_cache=False,
    )
    base_dataset.load_name_mapping({"diagnoses_icd": "d_icd_diagnoses", "procedures_icd": "d_icd_procedures"})
    if not os.path.exists(f"export/{dataset}/mortality.pkl"):
        target_dataset = MIMIC4Dataset(
            root="data/mimiciv/",
            tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
            code_mapping={
                "ICD9CM": "CCSCM",
                "ICD10CM": "CCSCM",
                "ICD9PROC": "CCSPROC",
                "ICD10PROC": "CCSPROC",
                "NDC": ("ATC", {"target_kwargs": {"level": 3}}),  # graphcare
            },
            dev=False,
            refresh_cache=False,
        )
    else:
        target_dataset = None
base_dataset.convert_name_in_patient_dict()
base_dataset.stat()
task_fn = mortality_prediction_mimic3_fn if dataset == "mimiciii" else mortality_prediction_mimic4_fn
target_dataset = set_task(target_dataset, task_fn, dataset, "mortality", ignored_keys=["input_ids"])
visit_ids = []
for visit in target_dataset:
    visit_ids.append(visit["visit_id"])

if dataset == "mimiciii":
    sample_dataset = set_task(base_dataset, mimic3_fn_for_generation, dataset, "preprocess", ignored_keys=["diagnosis", "drug_days"], ids=visit_ids)
else:
    sample_dataset = set_task(base_dataset, mimic4_fn_for_generation, dataset, "preprocess", ignored_keys=["diagnosis", "drug_days"], ids=visit_ids)
sample_dataset.stat()
np.random.seed(seed)
patient_indx = list(sample_dataset.patient_to_index.keys())
num_patients = len(patient_indx)
np.random.shuffle(patient_indx)
train_patient_indx = patient_indx[: int(num_patients * 0.8)]
train_index = list(chain(*[sample_dataset.patient_to_index[i] for i in train_patient_indx]))

def copy_current_script(destination_folder):
    current_script = os.path.abspath(__file__)
    destination_file = os.path.join(destination_folder, os.path.basename(current_script))
    shutil.copy(current_script, destination_file)
copy_current_script(export_dir)

data = generate_prompt(cluster=use_cluster)
llm = LLM(model_id, tensor_parallel_size=gpus, trust_remote_code=True, load_format='auto', 
          gpu_memory_utilization=0.9, enforce_eager=False, disable_custom_all_reduce=True)
sampling_params = SamplingParams(n=n, max_tokens=1024, top_p=0.95)

perplexities = {}
for i in tqdm(range(0, len(data['name']), batch_size), total=math.ceil(len(data['name']) / batch_size)):
    if 'text' in data:
        prompts = data['text'][i : i + batch_size]
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    else:
        prompt_token_ids = data['input_ids'][i : i + batch_size]
        outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params, use_tqdm=False)
    names = data['name'][i : i + batch_size]
    for j in range(len(names)):
        perplexities[names[j]] = {}
        for k in range(len(outputs[j].outputs)):
            perplexity = math.exp(-outputs[j].outputs[k].cumulative_logprob / len(outputs[j].outputs[k].token_ids))
            perplexities[names[j]][k] = perplexity
            file_thread = threading.Thread(target=file_processing, args=(os.path.join(export_dir, f'{names[j]}-{k}.txt'), outputs[j].outputs[k].text.strip()))
            file_thread.start()
        
with open(os.path.join(export_dir, 'perplexity.pkl'), 'wb') as f:
    pickle.dump(perplexities, f)
    