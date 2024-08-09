# Code of IntelliCare

Official implementation of IntelliCare: Improving Healthcare Analysis with Patient-Level Knowledge from Large Language Models

## Data Preparation
- MIMIC-III dataset: https://physionet.org/content/mimiciii/1.4/
- MIMIC-IV dataset: https://physionet.org/content/mimiciv/2.2/

You need to follow the instructions on the PhysioNet website to access the data. After downloading the data, please put the data in the `data` folder.

## LLM Preparation
You can download the pre-trained LLMs from the Hugging Face model hub. The LLMs that we used in the paper are:
- Meta-Llama-3-8B-Instruct: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
- Mistral-7B-Instruct-v0.3: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3

To extract the embeddings from the LLMs, you need to download `gte-large-en-v1.5` in https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5.

## Training and Evaluation
1. Install the required packages by running `pip install -r requirements.txt`. If you encounter the version conflict, you can install the pyhealth package by `pip install pyhealth --no-dependencies`.

2. Generate analyses from LLMs by running the `generate_analysis.py` file. You can modify the `model_id` to select the model that you want to use. If you want to disable the information from cohorts, you can set `use_cluster=False`. For the MIMIC-III dataset, this step will take around 1 hours for clustering and 5 hours for LLM inference if you choose `n=8` (`n` is the number of analyses) on 4*3090 GPUs.

3. Extract the embeddings from the LLMs by running the `extract_embeddings.py` file.

4. Train the EHR encoder. The training code is in the `train_encoder.py` folder. You can modify `--backbone` in the choices to select the model that you want to train. We put part of the data files of GraphCare we used in `resources/graphcare/`, and you need to put other files into this folder if you select [GraphCare](https://github.com/pat-jj/GraphCare) as the EHR encoder.

5. Train the IntelliCare model. The training code is in the `train_intellicare.py` folder. You can modify `--backbone` to select the EHR encoder.

Note that the training of `GraphCare` is very slow and consumes lots of CPUs since constructing mini-batch via its dataloader is complex, about 4 hours for one experiment.

We put a script `run.sh` to run the training pipeline if you need. You can modify our script to select the model or hyperparameters that you want to use.
