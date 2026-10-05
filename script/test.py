import os
import warnings
import pickle as pkl
import torch
import torch.nn.functional as F
from config import Config
from model import GOFormer
from data_loader import create_data_loaders, load_go_graph
from utils import evaluate_propagated, compute_fmax_and_aupr_propagated
import argparse

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.metrics')

def load_pkl(path):
    with open(path, "rb") as f:
        return pkl.load(f)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="bp",
        choices=["bp", "cc", "mf"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    Config.set_task(args.task)
    config = Config()
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    train_loader, val_loader, test_loader = create_data_loaders(config)
    go_graph = load_go_graph(config.GO_GRAPH_PATH).to(device)
    go_graph_data = load_pkl(config.GO_GRAPH_PATH)
    idx_to_go = go_graph_data.get('idx_to_go', {})
    go_obo_path = getattr(config, 'GO_OBO_PATH', '../data/go.obo')
    go_data = load_pkl(config.none_embed_pkl_path)
    pretrained_numpy  = go_data['pretrained_embeddings']
    pretrained_tensor = torch.FloatTensor(pretrained_numpy[:, :config.HIDDEN_DIM])
    go_data1 = load_pkl(config.weitiao_embed_pkl_path)
    pretrained_numpy1 = go_data1['pretrained_embeddings']
    reduced_embeddings1 = pretrained_numpy1[:, :config.HIDDEN_DIM]
    pretrained_tensor1 = torch.FloatTensor(reduced_embeddings1)
    pretrained_tensor = (pretrained_tensor + pretrained_tensor1)
    pretrained_tensor = F.normalize(pretrained_tensor, p=2, dim=-1)

    model = GOFormer(
        config=config,
        go_graph=go_graph,
        pretrained_label_embeds=pretrained_tensor,

    ).to(device)

    all_probs_avg = None
    all_labels = None
    all_protein_ids = None

    for epoch in range(0, 3):
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth')
        if not os.path.exists(ckpt_path):
            print(f"  Checkpoint not found: {ckpt_path}, skipping...")
            continue
        checkpoint = torch.load(
            ckpt_path,
            weights_only=False
        )
        print(f"Loading checkpoint from {ckpt_path} for evaluation...")
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
        all_probs, all_labels, all_protein_ids = evaluate_propagated(model, test_loader, device,train_loader)
        if all_probs_avg is None:
            all_probs_avg = all_probs
            all_labels = all_labels
            all_protein_ids = all_protein_ids
        else:
            all_probs_avg += all_probs
    metrics = compute_fmax_and_aupr_propagated(
            all_probs_avg/3.0, all_labels, all_protein_ids, idx_to_go, go_obo_path, ont=args.task
        )
    print(f"\n{'='*60}")
    print(f"  Test Fmax      : {metrics['fmax']:.4f}  (threshold={metrics['threshold']:.2f})")
    print(f"  Test AUPR      : {metrics['aupr']:.4f}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
