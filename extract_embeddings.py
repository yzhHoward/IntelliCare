import argparse
import math
import os
import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from pyhealth.datasets import collate_fn_dict
from transformers import AutoTokenizer, AutoModel

from utils.mimic import MIMIC3Dataset, MIMIC4Dataset
from utils.tasks import set_task, mortality_prediction_mimic3_fn, mortality_prediction_mimic4_fn, read_text
from utils.utils import set_seed


class TextEncoder(nn.Module):
    def __init__(self, tokenizer, encoder):
        super().__init__()
        self.tokenizer = tokenizer
        self.encoder = encoder
        
    def forward(self, text):
        text = [item for sublist in text for item in sublist]
        B = len(text)
        if self.tokenizer is None:
            input_ids = torch.tensor(input_ids)
        else:
            current_rank = torch.cuda.current_device()
            world_size = torch.cuda.device_count()
            b = math.ceil(B / world_size)
            input = self.tokenizer.batch_encode_plus(text[b * current_rank : b * (current_rank + 1)], padding=True, truncation=True, max_length=8192)
        text_features = self.encoder(torch.tensor(input["input_ids"], device='cuda'), torch.tensor(input["attention_mask"], device='cuda'))[0][:, 0]
        return text_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="./gte-large-en-v1.5")
    parser.add_argument("--dataset", type=str, default='mimiciii', choices=['mimiciii', 'mimiciv'])
    parser.add_argument("--task", type=str, default='mortality', choices=['readmission', 
        'mortality', 'length_of_stay'])
    parser.add_argument("--text_path", type=str, default='export/mimiciii/llama-cluster-8x/')
    parser.add_argument("--text_num", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()
    set_seed(args.seed)
    if os.path.exists(f"export/{args.dataset}/{args.task}.pkl"):
        base_dataset = None
    elif args.dataset == "mimiciii":
        base_dataset = MIMIC3Dataset(
            root="data/mimiciii/",
            tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
            code_mapping={"ICD9CM": "CCSCM", "ICD9PROC": "CCSPROC", "NDC": ("ATC", {"target_kwargs": {"level": 3}})},
            dev=False,
            refresh_cache=False,
        )
    else:
        base_dataset = MIMIC4Dataset(
            root="data/mimiciv/",
            tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
            code_mapping={
                "ICD9CM": "CCSCM",
                "ICD10CM": "CCSCM",
                "ICD9PROC": "CCSPROC",
                "ICD10PROC": "CCSPROC",
                "NDC": ("ATC", {"target_kwargs": {"level": 3}}),
            },
            dev=False,
            refresh_cache=False,
        )

    # STEP 2: set task
    if args.text_num != 0:
        with open(os.path.join(args.text_path, 'perplexity.pkl'), 'rb') as f:
            perplexities = pickle.load(f)
    else:
        perplexities = None
    sample_dataset = set_task(base_dataset, mortality_prediction_mimic3_fn if args.dataset == "mimiciii" else mortality_prediction_mimic4_fn, args.dataset, args.task, ignored_keys=["input_ids"])
    sample_dataset.stat()
    sample_dataset = read_text(sample_dataset, args.text_path, args.text_num, perplexities, return_text=True)
    
    sampler = SequentialSampler(sample_dataset)
    dataloader = DataLoader(sample_dataset, batch_size=args.batch_size // args.text_num if args.text_num != 0 else args.batch_size,
                            sampler=sampler, collate_fn=collate_fn_dict)

    # STEP 3: define model
    text_encoder = TextEncoder(AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True), AutoModel.from_pretrained(args.model_id, trust_remote_code=True)).cuda()
    text_encoder = nn.DataParallel(text_encoder)
    embeddings = {}
    for batch in tqdm(dataloader):
        with torch.no_grad():
            embedding = text_encoder(batch['text']).cpu()
        if args.text_num != 0:
            embedding = embedding.reshape(len(batch['text']), args.text_num, -1)
            for i in range(len(batch['text'])):
                embeddings[batch['visit_id'][i]] = {}
                for j in range(args.text_num):
                    embeddings[batch['visit_id'][i]][j] = embedding[i, j].tolist()
        else:
            for i in range(len(batch['text'])):
                embeddings[batch['visit_id'][i]] = embedding[i].tolist()
    with open(os.path.join(args.text_path, 'embedding.pkl'), 'wb') as f:
        pickle.dump(embeddings, f)
    
    