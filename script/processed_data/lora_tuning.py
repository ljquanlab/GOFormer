# import pickle
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# import torch
# from torch.utils.data import DataLoader

# from sentence_transformers import SentenceTransformer, InputExample, losses
# # ==========================================
# # [新增] 导入 LoRA 相关的 PEFT 库
# # ==========================================
# from peft import LoraConfig, get_peft_model 

# def main():
#     # ==========================================
#     # 1. 配置路径与超参数
#     # ==========================================
#     input_data_path = '/data01/zce/MYDataset/data/cc/my_qwen3_finetune_pairs_cc.pkl'
    
#     # [修改] 切换为 4B 模型
#     base_model_name = 'Qwen/Qwen3-Embedding-4B' 
    
#     # 微调后模型权重的保存目录
#     output_model_dir = './mydata/qwen3_finetuned_cc'
    
#     # 最终输出给主模型的特征文件路径
#     final_embeddings_path = './mydata/pdb_go_pretrained_llm_cc.pkl'
    
#     # 4B 模型建议 batch_size 先设为 4，显存如果还很富余可以尝试 8
#     batch_size = 4  
#     num_epochs = 3   

#     # ==========================================
#     # 2. 加载训练数据
#     # ==========================================
#     print(f"📦 正在加载训练数据包: {input_data_path} ...")
#     with open(input_data_path, 'rb') as f:
#         data = pickle.load(f)
        
#     train_pairs = data['positive_pairs']             # [("子", "父"), ...]
#     target_list = data['target_list']             # 你的原始标签列表 (保留了 alt_id，保证顺序)
#     go_texts_dict = data['go_texts_dict']         # ID -> 文本 的映射字典

#     print(f"✅ 成功加载 {len(train_pairs)} 对正样本，准备组装 DataLoader。")

#     # ==========================================
#     # 3. 构造 InputExample (注入指令)
#     # ==========================================
#     instruction = "Instruct: Find the parent Gene Ontology term for this biological process.\nQuery: "
    
#     train_examples = []
#     for child_text, parent_text in train_pairs:
#         anchor = instruction + child_text
#         positive = parent_text
#         train_examples.append(InputExample(texts=[anchor, positive]))

#     train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

#     # ==========================================
#     # 4. 加载模型并注入 LoRA (核心修改区)
#     # ==========================================
#     print(f"🤖 正在加载基础模型: {base_model_name} ...")
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     # [新增] 必须使用 bfloat16 精度加载，否则 4B 模型容易撑爆 24G 显存
#     model = SentenceTransformer(
#         base_model_name, 
#         trust_remote_code=True, 
#         device=device,
#         model_kwargs={"dtype": torch.bfloat16} 
#     )

#     print("🔧 正在注入 LoRA 适配器...")
#     # [新增] 配置 LoRA 参数，只微调注意力机制等核心层
#     lora_config = LoraConfig(
#         r=16,
#         lora_alpha=32,
#         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
#         lora_dropout=0.05,
#         bias="none",
#         task_type="FEATURE_EXTRACTION"
#     )

#     # [新增] 找到 SentenceTransformer 底层的 HuggingFace 模型并套上 LoRA
#     model[0].auto_model = get_peft_model(model[0].auto_model, lora_config)
#     model[0].auto_model.print_trainable_parameters() # 会在终端打印出只微调极少量的参数

#     # 使用对比学习最强 Loss 函数
#     train_loss = losses.MultipleNegativesRankingLoss(model=model)

#     # ==========================================
#     # 5. 启动微调
#     # ==========================================
#     print(f"\n🚀 开始对比学习微调 (Epochs: {num_epochs}, Batch Size: {batch_size}) ...")
#     warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
    
#     model.fit(
#         train_objectives=[(train_dataloader, train_loss)],
#         epochs=num_epochs,
#         warmup_steps=warmup_steps,
#         weight_decay=0.01,
#         show_progress_bar=True,
#         output_path=output_model_dir,
#         save_best_model=True
#     )
#     print(f"✅ 微调完成！模型权重已保存至: {output_model_dir}")

#     # ==========================================
#     # 6. 终极步骤：提取特征并完美对齐你的标签列表
#     # ==========================================
#     print("\n✨ 正在使用微调后的模型提取专属特征矩阵...")
    
#     # 将模型显式切换为评估模式 (由于带着 LoRA 权重，这一步在内存中直接进行编码)
#     model.eval()

#     texts_to_encode = [go_texts_dict[go_id] for go_id in target_list]
    
#     embeddings = model.encode(
#         texts_to_encode, 
#         batch_size=32, # 推理时不再需要存梯度，batch_size 可以适当开大
#         show_progress_bar=True,
#         normalize_embeddings=True, 
#         convert_to_tensor=True
#     )
    
#     # 由于使用 bfloat16 训练，提出来的特征可能是 bfloat16 格式，转为 float32 更通用
#     embeddings_np = embeddings.cpu().float().numpy() 
    
#     # 组装给主模型的最终格式
#     final_output = {
#         'idx_to_go': target_list,                
#         'pretrained_embeddings': embeddings_np   
#     }
    
#     os.makedirs(os.path.dirname(final_embeddings_path), exist_ok=True)
#     with open(final_embeddings_path, 'wb') as f:
#         pickle.dump(final_output, f)
        
#     print(f"\n🎉 大功告成！")
#     print(f"📊 特征矩阵维度: {embeddings_np.shape}")
#     print(f"💾 特征已保存至: {final_embeddings_path}")

# if __name__ == "__main__":
#     main()



    # ==========================================
    # 6. 终极步骤：提取特征 (仅主进程执行)
    # ==========================================
    # ⚠️ 多卡环境下，必须限制只有 0号卡 去写文件，否则会造成文件读写冲突报错
    # if is_main_process:
    #     print("\n✨ 正在使用微调后的模型提取专属特征矩阵...")
    #     model.eval()

    #     texts_to_encode = [go_texts_dict[go_id] for go_id in target_list]
        
    #     # 将提取转移到不求梯度的环境中
    #     with torch.no_grad():
    #         embeddings = model.encode(
    #             texts_to_encode, 
    #             batch_size=32, 
    #             show_progress_bar=True,
    #             normalize_embeddings=True, 
    #             convert_to_tensor=True
    #         )
        
    #     embeddings_np = embeddings.cpu().float().numpy() 
        
    #     final_output = {
    #         'idx_to_go': target_list,                
    #         'pretrained_embeddings': embeddings_np   
    #     }
        
    #     os.makedirs(os.path.dirname(final_embeddings_path), exist_ok=True)
    #     with open(final_embeddings_path, 'wb') as f:
    #         pickle.dump(final_output, f)
            
    #     print(f"\n🎉 大功告成！")
    #     print(f"📊 特征矩阵维度: {embeddings_np.shape}")
    #     print(f"💾 特征已保存至: {final_embeddings_path}")


import pickle
import os
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0,2' 
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from datasets import Dataset 
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.trainer import SentenceTransformerTrainer
from peft import LoraConfig, get_peft_model 

def main():

    input_data_path = '/data01/zce/qwen3-4B/ensemble_tune/mf_positive_pairs.pkl'
    base_model_name = 'Qwen/Qwen3-Embedding-4B' 
    output_model_dir = './weitiao/qwen3_finetuned_mf'
    per_device_batch_size = 4  
    num_epochs = 3   
    is_main_process = int(os.environ.get("LOCAL_RANK", 0)) == 0
    
    if is_main_process:
        print(f"📦 正在加载训练数据包: {input_data_path} ...")
        
    with open(input_data_path, 'rb') as f:
        data = pickle.load(f)
        
    train_pairs = data['positive_pairs']                 

    if is_main_process:
        print(f"✅ 成功加载 {len(train_pairs)} 对正样本。")

    instruction = "Instruct: Find the parent Gene Ontology term for this molecular function.\nQuery: "
    
    split_point = int(len(train_pairs) * 0.9)  # 90%训练，10%验证
    train_subset = train_pairs[:split_point]
    eval_subset = train_pairs[split_point:]

    # 分别构造 Dataset
    train_dataset = Dataset.from_dict({
        "anchor": [instruction + pair[0] for pair in train_subset],
        "positive": [pair[1] for pair in train_subset]
    })
    eval_dataset = Dataset.from_dict({
        "anchor": [instruction + pair[0] for pair in eval_subset],
        "positive": [pair[1] for pair in eval_subset]
    })



    # ==========================================
    # 4. 加载模型并注入 LoRA
    # ==========================================
    if is_main_process:
        print(f"🤖 正在加载基础模型: {base_model_name} ...")
        
    # 不需要显式设定 device='cuda'，Trainer 会自动处理多卡分配
    model = SentenceTransformer(
        base_model_name, 
        trust_remote_code=True, 
        model_kwargs={"dtype": torch.bfloat16} # 保持 bf16
    )

    if is_main_process:
        print("🔧 正在注入 LoRA 适配器...")
        
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # 只微调注意力层
        lora_dropout=0.05,
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )

    model[0].auto_model = get_peft_model(model[0].auto_model, lora_config)
    
    if is_main_process:
        model[0].auto_model.print_trainable_parameters() 

    # 损失函数
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    if is_main_process:
        print(f"\n🚀 开始对比学习微调 (Epochs: {num_epochs}, Per Device Batch Size: {per_device_batch_size}) ...")
    
    args = SentenceTransformerTrainingArguments(
        output_dir=output_model_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        
        # ========== 以下为新增的评估相关参数 ==========
        eval_strategy="steps",          # 按步数评估
        eval_steps=50,                  # 每 50 步计算一次 eval_loss
        save_strategy="steps",          # 保存策略必须与评估策略对齐
        save_steps=50,                  # 每 50 步保存一次 checkpoint
        load_best_model_at_end=True,    # 训练结束后自动加载验证集表现最好的模型
        metric_for_best_model="eval_loss", # 用验证集损失作为最佳模型评判标准
        greater_is_better=False,        # Loss 越小越好
        # =============================================
        
        warmup_ratio=0.1,
        bf16=True,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=10,
        dataloader_drop_last=True,
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # 新增
        loss=train_loss,
    )

    # 启动多卡微调
    trainer.train()
    
    if is_main_process:
        trainer.save_model(output_model_dir)
        print(f"✅ 微调完成！模型权重已保存至: {output_model_dir}")

if __name__ == "__main__":
    main()