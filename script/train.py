import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import numpy as np
from config import Config
from model import GOFormer
from loss import get_loss_function
from data_loader import create_data_loaders, load_go_graph
from utils import print_metrics
import torch.nn.functional as F
import logging
import warnings
import pickle as pkl
import random
from datetime import datetime
import argparse
from utils import evaluate_propagated, compute_fmax_and_aupr_propagated, compute_performance,save_results_to_pkl


def load_pkl(path):
    with open(path, "rb") as f:
        return pkl.load(f)

def log_set(log_path):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.metrics')
    os.makedirs(log_path, exist_ok=True)
    log_file = os.path.join(
        log_path,
        f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filename=log_file,
        filemode='w' 
    )
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seeds = [2025, 123, 2024, 3447, 2021]
def compute_contrastive_loss(seq_emb, label_emb, targets, temperature=0.07):
    seq_emb = F.normalize(seq_emb, dim=-1)
    label_emb = F.normalize(label_emb, dim=-1)
    logits = torch.matmul(seq_emb, label_emb.t()) / temperature
    target_stnorm = targets / (targets.sum(dim=-1, keepdim=True) + 1e-8)
    loss_p2l = -(target_stnorm * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    logits_l2p = logits.t()
    target_l2p = targets.t()
    target_l2p_stnorm = target_l2p / (target_l2p.sum(dim=-1, keepdim=True) + 1e-8)
    loss_l2p = -(target_l2p_stnorm * F.log_softmax(logits_l2p, dim=-1)).sum(dim=-1).mean()
    return (loss_p2l + loss_l2p) / 2


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, go_graph=None):
    model.train()
    total_loss = 0
    all_predictions = []
    all_targets = []
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=True)
    for batch in pbar:
        embeddings = batch['embedding'].to(device)
        labels = batch['labels'].to(device)
        fixed_null_input = batch['fixed_null_input'].to(device).float()
        struct = batch['struct_embedding'].to(device)
        optimizer.zero_grad()
        logits, seq_feats, label_feats, logits_dense, logits_graph, _ = model(embeddings,fixed_null_input=fixed_null_input,struct=struct,return_features=True)
        loss = criterion(logits, labels)
        logits_dense_loss = criterion(logits_dense, labels)
        logits_graph_loss = criterion(logits_graph, labels)
        loss_cl = compute_contrastive_loss(seq_feats, label_feats, labels)
        t_loss = loss + logits_dense_loss*0.5 + logits_graph_loss*0.5 + loss_cl
        t_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += t_loss.item()
        all_predictions.append(torch.sigmoid(logits).detach())
        all_targets.append(labels.detach())
        pbar.set_postfix({'loss': t_loss.item()})
    avg_loss = total_loss / len(train_loader)
    return avg_loss

@torch.no_grad()
def evaluate(model, val_loader, criterion, device, epoch,  phase="Val", go_graph=None):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []

    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [{phase}]")
    for batch in pbar:
        embeddings = batch['embedding'].to(device)
        labels = batch['labels'].to(device)
        fixed_null_input = batch['fixed_null_input'].to(device).float()
        struct = batch['struct_embedding'].to(device)
        logits, seq_feats, label_feats,logits_dense, logits_graph, _ = model(embeddings,fixed_null_input=fixed_null_input,struct=struct,return_features=True)
        loss = criterion(logits, labels)
        logits_dense_loss = criterion(logits_dense, labels)
        logits_graph_loss = criterion(logits_graph, labels)
        loss_cl = compute_contrastive_loss(seq_feats, label_feats, labels)
        t_loss = loss + logits_dense_loss*0.5 + logits_graph_loss*0.5 + loss_cl
        total_loss += t_loss.item()
        all_predictions.append(torch.sigmoid(logits))
        all_targets.append(labels)
        pbar.set_postfix({'loss': t_loss.item()})
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = compute_performance(all_targets, all_predictions)
    avg_loss = total_loss / len(val_loader)
    return avg_loss, metrics

def save_checkpoint(model, optimizer, epoch, metrics, config, filename, seed):
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics.item() if hasattr(metrics, 'item') else metrics,
        'config': config
    }
    path = os.path.join(config.CHECKPOINT_DIR, filename)
    torch.save(checkpoint, path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="bp",
        choices=["bp", "cc", "mf"],
    )
    return parser.parse_args()

def main(seed=2025):
    args = parse_args()
    Config.set_task(args.task)
    config = Config()
    log_set(log_path=config.LOG_DIR)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_seed(seed)
    train_loader, val_loader, test_loader = create_data_loaders(config)
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    logging.debug(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    go_graph = load_go_graph(config.GO_GRAPH_PATH).to(device)
    go_data = load_pkl(config.none_embed_pkl_path)
    pretrained_numpy = go_data['pretrained_embeddings']
    reduced_embeddings = pretrained_numpy[:, :config.HIDDEN_DIM]
    pretrained_tensor = torch.FloatTensor(reduced_embeddings) 
    go_data1 = load_pkl(config.weitiao_embed_pkl_path)
    pretrained_numpy1 = go_data1['pretrained_embeddings']
    reduced_embeddings1 = pretrained_numpy1[:, :config.HIDDEN_DIM]
    pretrained_tensor1 = torch.FloatTensor(reduced_embeddings1)
    pretrained_tensor = (pretrained_tensor + pretrained_tensor1)
    pretrained_tensor = F.normalize(pretrained_tensor, p=2, dim=-1)
    model = GOFormer(
        config=config, 
        go_graph=go_graph, 
        pretrained_label_embeds=pretrained_tensor
    )
    model = model.to(device)
    criterion = get_loss_function(config, go_graph=None)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    patience_counter = 0
    low_loss = float('inf')
    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,go_graph
        )
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, device, epoch, phase="Val",go_graph=go_graph
        )
        logging.debug(f"\nEpoch {epoch}/{config.NUM_EPOCHS}")
        logging.debug(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print_metrics(val_metrics, prefix="Val")
        print(f'Fmax: {val_metrics["fmax"]:.4f}, AUPR: {val_metrics["aupr"]:.4f}, threshold: {val_metrics["threshold"]:.4f}')
        if val_loss < low_loss:
            low_loss = val_loss
            print(f"New best validation loss: {low_loss:.4f} at epoch {epoch}, saving model...")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter < 3:
                save_checkpoint(model, optimizer, epoch, val_metrics, config, f'checkpoint_epoch_{patience_counter}.pth', seed)
                # save_checkpoint(model, optimizer, epoch, val_metrics, config, f'checkpoint_epoch_{epoch-1}.pth', seed)
        if patience_counter >= config.EARLY_STOP_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch} with best validation loss: {low_loss:.4f}")
            break
    all_probs_avg = None
    all_labels = None
    all_protein_ids = None
    
    for i in range(3):
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, f'checkpoint_epoch_{i}.pth')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path,weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            all_probs, all_labels, all_protein_ids = evaluate_propagated(model, test_loader, device,train_loader)
            if all_probs_avg is None:
                all_probs_avg = all_probs
                all_labels = all_labels
                all_protein_ids = all_protein_ids
            else:
                all_probs_avg += all_probs

    with open(config.GO_GRAPH_PATH, 'rb') as f:
        go_graph_data = pkl.load(f)

    idx_to_go = go_graph_data.get('idx_to_go', {})
    metrics = compute_fmax_and_aupr_propagated(
        all_probs_avg/3.0, all_labels, all_protein_ids, idx_to_go, config.GO_OBO_PATH, ont=args.task
    )
    # save result
    save_results_to_pkl(
        all_probs_avg/3.0, all_labels, all_protein_ids, idx_to_go, metrics, output_path=config.TEST_RESULTS_PKL)

    print(f"\n{'='*60}")
    print(f"  Test Fmax      : {metrics['fmax']:.4f}  (threshold={metrics['threshold']:.2f})")
    print(f"  Test AUPR      : {metrics['aupr']:.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    for i in range(5):
        seed = seeds[i]
        print(f"Running with seed: {seed}")
        main(seed=seed) 
        torch.cuda.empty_cache()


