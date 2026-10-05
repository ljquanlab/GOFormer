import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from config import Config
import dgl

class ProteinDataset(Dataset):
    def __init__(self, embedding_dict, labels_dict, protein_ids,struct_embeddings=None):
        """
        Args:
            embedding_dict: {protein_id: embedding_vector} 预计算的ESM embedding
            labels_dict: {protein_id: label_vector} 多标签向量
            protein_ids: list of protein IDs
            struct_embeddings: {protein_id: struct_embedding_vector} 结构特征字典
        """
        self.embedding_dict = embedding_dict
        self.labels_dict = labels_dict
        self.protein_ids = protein_ids
        self.struct_embeddings = struct_embeddings

    def __len__(self):
        return len(self.protein_ids)

    def __getitem__(self, idx):
        protein_id = self.protein_ids[idx]

        embedding = self.embedding_dict[protein_id]
        labels = self.labels_dict[protein_id]
        if self.struct_embeddings is not None:
            struct_embedding = self.struct_embeddings.get(protein_id, np.zeros(Config.STRUCT_DIM, dtype=np.float32))
        else:
            struct_embedding = np.zeros(Config.STRUCT_DIM, dtype=np.float32)
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding, dtype=torch.float32)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.float32)
        fixed_null_input = np.zeros(Config.null_input_dim, dtype=np.float32)
        if not isinstance(fixed_null_input, torch.Tensor):
            fixed_null_input = torch.tensor(fixed_null_input, dtype=torch.float32)
        if not isinstance(struct_embedding, torch.Tensor):
            struct_embedding = torch.tensor(struct_embedding, dtype=torch.float32)
        if struct_embedding.shape[0] == 512:
            struct_embedding = struct_embedding.unsqueeze(0)
        return {
            'protein_id': protein_id,
            'embedding': embedding,
            'labels': labels,
            'fixed_null_input': fixed_null_input,
            'struct_embedding': struct_embedding
        }

def load_embeddings(pkl_path):
    with open(pkl_path, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings


def load_go_graph(pkl_path):
    with open(pkl_path, 'rb') as f:
        go_data = pickle.load(f)
    if 'edges' in go_data:
        edges = go_data['edges']
        num_nodes = go_data['num_nodes']
        if len(edges) == 0:
            g = dgl.graph(([], []), num_nodes=num_nodes)
        else:
            src_nodes = [e[0] for e in edges] 
            dst_nodes = [e[1] for e in edges]
            g = dgl.graph((src_nodes, dst_nodes), num_nodes=num_nodes)
        g = dgl.add_self_loop(g)
        return g
    elif 'dgl_graph' in go_data:
        return go_data['dgl_graph']
    else:
        raise ValueError("Invalid GO graph format.")

def create_data_loaders(config):
    train_embeddings = load_embeddings(config.SEQ_TRAIN_EMBEDDING_PKL_PATH)
    valid_embeddings = load_embeddings(config.SEQ_VALID_EMBEDDING_PKL_PATH)
    test_embeddings = load_embeddings(config.SEQ_TEST_EMBEDDING_PKL_PATH)
    train_struct_embeddings = torch.load(config.TRAIN_STRUCT_EMBEDDING_PATH, map_location='cpu', weights_only=True)
    valid_struct_embeddings = torch.load(config.VALID_STRUCT_EMBEDDING_PATH, map_location='cpu', weights_only=True)
    test_struct_embeddings = torch.load(config.TEST_STRUCT_EMBEDDING_PATH, map_location='cpu', weights_only=True)
    with open(config.TRAIN_DATA_PATH, 'rb') as f:
        train_data_labels = pickle.load(f)
        train_protein_ids = list(train_data_labels.keys())
    with open(config.VAL_DATA_PATH, 'rb') as f:
        val_data_labels = pickle.load(f)
        val_protein_ids = list(val_data_labels.keys())
    with open(config.TEST_DATA_PATH, 'rb') as f:
        test_data_labels = pickle.load(f)
        test_protein_ids = list(test_data_labels.keys())
    train_dataset = ProteinDataset(
        embedding_dict=train_embeddings, 
        labels_dict=train_data_labels, 
        protein_ids=train_protein_ids,
        struct_embeddings=train_struct_embeddings,                 
    )
    val_dataset = ProteinDataset(
        embedding_dict=valid_embeddings, 
        labels_dict=val_data_labels, 
        protein_ids=val_protein_ids,
        struct_embeddings=valid_struct_embeddings,
    )
    test_dataset = ProteinDataset(
        embedding_dict=test_embeddings, 
        labels_dict=test_data_labels, 
        protein_ids=test_protein_ids,
        struct_embeddings=test_struct_embeddings,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, val_loader, test_loader




