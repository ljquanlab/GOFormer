import sys
import pickle
import numpy as np

# === 核心配置区 ===
# 建议使用溯源后的数据，保证标签树完整！
input_tsv = "train_labels_propagated.tsv"  
vocab_csv = "mf_terms_filter.csv"             # 刚才生成的 BP 字典
output_pkl = "train_mf_multihot.pkl"    # 最终输出的魔法文件
# ==================

def main():
    print(f"📖 步骤 1: 正在加载 BP 字典 {vocab_csv} ...")
    go_to_idx = {}
    cnt = 0
    try:
        with open(vocab_csv, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if "GO_ID" in line.upper() or line == "0" or "Unnamed" in line:
                    continue
                if ',' in line:
                    _, go_id = line.split(',', 1)
                    go_id = go_id.strip()
                else:
                    go_id = line
                
                go_to_idx[go_id] = cnt
                cnt += 1
    except FileNotFoundError:
        sys.exit(1)

    num_classes = len(go_to_idx)

    prot_to_multihot = {}
    try:
        with open(input_tsv, 'r', encoding='utf-8') as f:
            header = f.readline()
            for line in f:
                line = line.strip()
                if not line: continue
                cols = line.split('\t')
                prot_id = cols[0].strip()
                go_id = cols[1].strip()
                if go_id in go_to_idx:
                    if prot_id not in prot_to_multihot:
                        prot_to_multihot[prot_id] = np.zeros(num_classes, dtype=np.int8)
                    idx = go_to_idx[go_id]
                    prot_to_multihot[prot_id][idx] = 1
                    
    except Exception as e:
        print(f"解析 TSV 时出错: {e}")
        sys.exit(1)
    try:
        with open(output_pkl, 'wb') as f:
            pickle.dump(prot_to_multihot, f)
        print(f"最终数据概览: {len(prot_to_multihot)} 个蛋白质，每个蛋白质对应一个长度为 {num_classes} 的 Multi-hot 向量。")
    except Exception as e:
        print(f"保存 Pickle 文件时出错: {e}")

if __name__ == "__main__":
    main()