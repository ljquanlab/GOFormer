import argparse
import pickle as pkl

import esm
import numpy as np
import torch
from tqdm.auto import trange

# 允许加载 Namespace（部分环境下加载权重需要）
torch.serialization.add_safe_globals([argparse.Namespace])


def load_pkl(path):
    with open(path, "rb") as f:
        return pkl.load(f)


def save_pkl(path, data):
    with open(path, "wb") as f:
        pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)


def read_fasta(file_path):
    """返回 {pdb_chain_id: sequence}，ID 为 '>' 后第一个空白分隔字段（与 nrPDB 注释一致，如 11AS-A）。"""
    sequences = {}
    with open(file_path, "r", encoding="utf-8") as f:
        current_id = None
        current_seq = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(current_seq)
                header = line[1:].strip()
                current_id = header.split()[0] if header else header
                current_seq = []
            else:
                current_seq.append(line)
        if current_id is not None:
            sequences[current_id] = "".join(current_seq)
    return sequences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fasta",
        type=str,
        default="train_seqs.pkl",
        help="输入 pkl",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="train_embeddings.pkl",
        help="输出 {id: numpy.ndarray} 的 pickle",
    )
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument(
        "--device",
        type=str,
        default='cuda:3',
        help="如 cuda:0；默认自动选 cuda 或 cpu",
    )
    parser.add_argument("--max-len", type=int, default=1022, help="ESM2 截断长度")
    args = parser.parse_args()

    model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    model = model.eval()
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model = model.to(device)

    # protein_seqs = read_fasta(args.fasta)
    protein_seqs = load_pkl(args.fasta)
    batch_converter = alphabet.get_batch_converter()
    embedding_dict = {}

    id_seqs = []
    for k, v in protein_seqs.items():
        if len(v) > args.max_len:
            v = v[: args.max_len]
        id_seqs.append((k, v))

    batch_size = args.batch_size
    n_batch = (len(id_seqs) + batch_size - 1) // batch_size

    for batch_idx in trange(n_batch, desc="ESM2"):
        batch_data = id_seqs[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            result = model(
                batch_tokens,
                repr_layers=[36],
                return_contacts=False,
            )

        tokens_representation = result["representations"][36]

        for j in range(len(batch_labels)):
            token_len = batch_lens[j].item()
            # 去掉 BOS/EOS，对残基位置做 mean pooling
            embedding_dict[batch_labels[j]] = (
                tokens_representation[j, 1 : token_len - 1].mean(dim=0).cpu().numpy()
            )

    save_pkl(args.out, embedding_dict)
    print(f"Saved {len(embedding_dict)} embeddings -> {args.out}")


if __name__ == "__main__":
    main()
