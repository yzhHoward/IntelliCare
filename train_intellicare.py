import argparse
import os
import pickle
import torch
from torch.utils.data import DataLoader
from pyhealth.datasets import split_by_patient, collate_fn_dict
from pyhealth.models import AdaCare, StageNet, ConCare, GRASP

from models.safari import SAFARI
from models.graphcare import GraphCare, load_dataset
from models.intellicare import IntelliCare
from utils.mimic import MIMIC3Dataset, MIMIC4Dataset
from utils.trainer import Trainer
from utils.tasks import set_task, readmission_prediction_mimic3_fn, mortality_prediction_mimic3_fn, length_of_stay_prediction_mimic3_fn, readmission_prediction_mimic4_fn, mortality_prediction_mimic4_fn, length_of_stay_prediction_mimic4_fn, read_text
from utils.utils import set_seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='mimiciii', choices=['mimiciii', 'mimiciv'])
    parser.add_argument("--task", type=str, default='mortality', choices=['readmission', 
        'mortality', 'length_of_stay'])
    parser.add_argument("--text_path", type=str, default='export/mimiciii/llama-cluster-8x')
    parser.add_argument("--text_num", type=int, default=8)
    parser.add_argument("--backbone", type=str, default='concare', choices=['concare', 'grasp', 
                                                                        'adacare', 'safari', 'stagenet', 'graphcare'])
    parser.add_argument("--suffix", type=str, default='test')
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--finetune_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument("--finetune_lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--embedding_file", type=str, default='embedding.pkl')
    parser.add_argument("--text_dim", type=int, default=1024)
    args = parser.parse_args()
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

    if args.text_num != 0:
        with open(os.path.join(args.text_path, 'perplexity.pkl'), 'rb') as f:
            perplexities = pickle.load(f)
    else:
        perplexities = None
    if os.path.exists(os.path.join(args.text_path, args.embedding_file)):
        with open(os.path.join(args.text_path, args.embedding_file), 'rb') as f:
            text_embedding = pickle.load(f)
    else:
        text_embedding = None
    
    if args.task == "mortality":
        task_fn = mortality_prediction_mimic3_fn if args.dataset == "mimiciii" else mortality_prediction_mimic4_fn
        mode = "binary"
        monitor = "roc_auc"
        metrics = None
    elif args.task == "readmission":
        task_fn = readmission_prediction_mimic3_fn if args.dataset == "mimiciii" else readmission_prediction_mimic4_fn
        mode = "binary"
        monitor = "roc_auc"
        metrics = None
    elif args.task == "length_of_stay":
        task_fn = length_of_stay_prediction_mimic3_fn if args.dataset == "mimiciii" else length_of_stay_prediction_mimic4_fn
        mode = "multiclass"
        monitor = "roc_auc_weighted_ovr"
        metrics = ["roc_auc_weighted_ovr", "f1_weighted", "accuracy", "cohen_kappa"]
    
    sample_dataset = set_task(base_dataset, task_fn, args.dataset, args.task, ignored_keys=["input_ids"])
    sample_dataset = read_text(sample_dataset, args.text_path, args.text_num, perplexities, text_embedding)

    if args.backbone == "graphcare":
        args.batch_size = 16
        train_dataloader, val_dataloader, test_dataloader, node_emb, rel_emb, sample_dataset = load_dataset(args.dataset, args.task + "_finetune", sample_dataset, args.batch_size)
    else:
        train_dataset, val_dataset, test_dataset = split_by_patient(
            sample_dataset, [0.8, 0.1, 0.1], seed=2 if args.dataset == "mimiciii" else 1
        )
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_dict, drop_last=True if args.backbone == "grasp" else False)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn_dict, drop_last=True if args.backbone == "grasp" else False)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn_dict, drop_last=True if args.backbone == "grasp" else False)
    set_seed(args.seed)
    
    embedding_dim = 32
    hidden_dim = 128
    if args.backbone == "stagenet":
        hidden_dim = 32 * 3
        backbone = StageNet(sample_dataset, ["conditions", "procedures", "drugs"], "label", mode, embedding_dim=embedding_dim, chunk_size=32)
    elif args.backbone == "adacare":
        backbone = AdaCare(sample_dataset, ["conditions", "procedures", "drugs"], "label", mode,
            use_embedding=[True, True, True], embedding_dim=embedding_dim, hidden_dim=hidden_dim, dropout=0.5)
    elif args.backbone == "concare":
        backbone = ConCare(sample_dataset, ["conditions", "procedures", "drugs"], "label", mode,
            use_embedding=[True, True, True], embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    elif args.backbone == "grasp":
        backbone = GRASP(sample_dataset, ["conditions", "procedures", "drugs"], "label", mode,
            use_embedding=[True, True, True], embedding_dim=embedding_dim, hidden_dim=hidden_dim, cluster_num=4)
    elif args.backbone == "safari":
        backbone = SAFARI(sample_dataset, ["conditions", "procedures", "drugs"], "label", mode,
            use_embedding=[True, True, True], embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    elif args.backbone == "graphcare":
        args.finetune_lr = 1.25e-5
        args.finetune_epochs = 15
        backbone = GraphCare(sample_dataset, mode=mode, feature_keys=[0, 1], label_key="label",
            num_nodes=node_emb.shape[0], num_rels=rel_emb.shape[0],
            max_visit=sample_dataset[0]['visit_padded_node'].shape[0], embedding_dim=node_emb.shape[1],
            hidden_dim=hidden_dim, layers=1, dropout=0.5, decay_rate=0.01, node_emb=node_emb, rel_emb=rel_emb,
            patient_mode='joint', use_alpha=False, use_beta=False, use_edge_attn=True, gnn="BAT",
            freeze=False, attn_init=None)

    if os.path.exists(os.path.join('output', args.dataset + '_' + args.task, args.backbone + '-' + str(args.seed), 'best.ckpt')):
        backbone.load_state_dict(torch.load(os.path.join('output', args.dataset + '_' + args.task, args.backbone + '-' + str(args.seed), 'best.ckpt'), map_location='cpu')['state_dict'])
    else:
        raise ValueError("No pre-trained model found!")
    
    model = IntelliCare(backbone, sample_dataset, "label", mode, 
                  hidden_dim=hidden_dim, text_dim=args.text_dim, dropout=args.dropout).cuda()
    params = []
    for name, param in model.named_parameters():
        if 'backbone' not in name:
            params.append(param)
    optimizer = torch.optim.Adam(params, args.finetune_lr)
    trainer = Trainer(model=model, device='cuda', metrics=metrics, new_logging=True,
                    exp_name=args.dataset + '_' + args.task + '/' + args.backbone + args.suffix + '-' + str(args.seed))
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        epochs=args.finetune_epochs,
        optimizer=optimizer,
        monitor=monitor,
    )
    