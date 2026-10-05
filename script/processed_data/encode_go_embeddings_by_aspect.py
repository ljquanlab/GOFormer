import argparse
import pickle
from pathlib import Path
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
from sentence_transformers import SentenceTransformer


def encode_one(input_pkl, model, output_pkl, batch_size):
    with open(input_pkl, "rb") as f:
        data = pickle.load(f)

    target_list = data["target_list"]
    go_texts_dict = data["go_texts_dict"]

    final_target_list = []
    texts = []

    for go_id in target_list:
        if go_id not in go_texts_dict:
            print(f"Warning: {go_id} not found in go_texts_dict. Skip.")
            continue
        final_target_list.append(go_id)
        texts.append(go_texts_dict[go_id])

    print(f"Encoding {len(texts)} labels from {input_pkl}")

    with torch.no_grad():
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_tensor=True,
        )

    embeddings_np = embeddings.cpu().float().numpy()

    output = {
        "idx_to_go": final_target_list,
        "pretrained_embeddings": embeddings_np,
    }

    output_pkl = Path(output_pkl)
    output_pkl.parent.mkdir(parents=True, exist_ok=True)

    with open(output_pkl, "wb") as f:
        pickle.dump(output, f)

    print(f"Saved: {output_pkl}")
    print(f"Shape: {embeddings_np.shape}")


def main():
    parser = argparse.ArgumentParser()
    tt = 'cc'
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Embedding-4B")
    # parser.add_argument("--model", type=str, default=f"/data01/zce/qwen3-4B/weitiao/qwen3_finetuned_{tt}")

    parser.add_argument("--mf-pkl", type=str, default=f'/data01/zce/PDBdataset/data/nrPDB/{tt}_positive_pairs.pkl')
    # parser.add_argument("--bp-pkl", type=str, default='./bp_positive_pairs.pkl')
    # parser.add_argument("--cc-pkl", type=str, default='./cc_positive_pairs.pkl')
    parser.add_argument("--out-dir", type=str, default="/data01/zce/PDBdataset/data/nrPDB/none")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    print(f"Loading finetuned model: {args.model}")
    model = SentenceTransformer(
        args.model,
        trust_remote_code=True,
        model_kwargs={"dtype": torch.bfloat16},
    )

    model.eval()

    out_dir = Path(args.out_dir)

    encode_one(
        input_pkl=args.mf_pkl,
        model=model,
        output_pkl=out_dir / f"{tt}_go_pretrained_qwen3_from_all.pkl",
        batch_size=args.batch_size,
    )

    # encode_one(
    #     input_pkl=args.bp_pkl,
    #     model=model,
    #     output_pkl=out_dir / "bp_go_pretrained_qwen3_from_all.pkl",
    #     batch_size=args.batch_size,
    # )

    # encode_one(
    #     input_pkl=args.cc_pkl,
    #     model=model,
    #     output_pkl=out_dir / "cc_go_pretrained_qwen3_from_all.pkl",
    #     batch_size=args.batch_size,
    # )


if __name__ == "__main__":
    main()