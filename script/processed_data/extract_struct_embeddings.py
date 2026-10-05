import os
import torch
import esm
import pandas as pd
import esm.inverse_folding.util as util
from tqdm import tqdm

def load_pkl(path):
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)

# ================= 1. 配置路径与环境 =================
pdb_path = '../pdb/train/'  # 修改为 valid 数据集的 PDB 文件夹
output_dir = './embeddings/'  # 新增：特征保存的文件夹
os.makedirs(output_dir, exist_ok=True)


protein_pkl = "./train_seqs.pkl"
protein_ids = list(load_pkl(protein_pkl).keys())


device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
print(f"当前使用的计算设备: {device}")

# ================= 2. 加载模型 =================
print("正在加载 ESM-IF1 模型...")
model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
model = model.to(device)
model.eval()

# 必须使用官方提供的转换器，它负责处理 numpy转tensor、Batch维度拼接、以及过滤 NaN 空缺坐标
batch_converter = util.CoordBatchConverter(alphabet) 

# ================= 3. 批量提取特征 =================
print(f"总计需要处理 {len(protein_ids)} 个蛋白质...")

struct_embeddings = {}  # 用于存储所有蛋白质的特征
struct_embeddings = torch.load("./embeddings/train_struct_embeddings.pt", weights_only=True)
print(f"已加载 {len(struct_embeddings)} 个已提取的特征，继续处理剩余蛋白质...")

for prot_id in tqdm(protein_ids):
    pdb_file = os.path.join(pdb_path, f"{prot_id}.pdb")
    output_file = os.path.join(output_dir, f"{prot_id}.pt")

    # 检查 PDB 是否存在
    if not os.path.exists(pdb_file):
        print(f"文件缺失: {pdb_file} ，跳过...")
        continue

    if prot_id in struct_embeddings:
        # print(f"特征已存在: {prot_id} ，跳过...")
        continue
    
    # 检查是否已经提取过 (断点续传)
    if os.path.exists(output_file):
        print(f"特征已存在: {output_file} ，跳过...")
        continue
    try:
        # 读取坐标
        coords, seq = util.load_coords(pdb_file, 'A')

        # 组装 batch 并进行标准化转换
        batch = [(coords, None, seq)]
        batched_coords, confidence, _, _, padding_mask = batch_converter(batch)

        # 将数据移动到 GPU 上
        batched_coords = batched_coords.to(device)
        padding_mask = padding_mask.to(device)
        if confidence is not None:
            confidence = confidence.to(device)

        # 提取结构特征
        with torch.no_grad():
            # encoder() 返回的是一个字典
            encoder_out = model.encoder(
                batched_coords, 
                encoder_padding_mask=padding_mask, 
                confidence=confidence
            )
            
            # 提取张量，Fairseq 架构输出形状为: [Length, Batch, Hidden_dim]
            hidden_states = encoder_out['encoder_out'][0] 
            
            # 把序列长度放在第一维：[Length, Batch, 512] -> [Length, 512] (因为 Batch=1)
            reps = hidden_states[1:-1, 0, :]
            
            # 在序列长度维度上求平均，得到全局特征 [1, 512]
            protein_embedding = reps.mean(dim=0).unsqueeze(0)
            

            struct_embeddings[prot_id] = protein_embedding

    except Exception as e:
        print(f"处理 {prot_id} 时发生错误: {e}")
        continue
torch.save(struct_embeddings, os.path.join(output_dir, "train_struct_embeddings.pt"))
print("全部处理完成！")