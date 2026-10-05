import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, deque
from config import Config as config
import dgl.nn.pytorch as dglnn



def extract_edges(go_graph):
    if go_graph is None:
        return []
    try:
        import dgl
        if isinstance(go_graph, dgl.DGLGraph):
            src, dst = go_graph.edges()
            pairs = list(zip(src.detach().cpu().tolist(), dst.detach().cpu().tolist()))
            return [(u, v) for u, v in pairs if u != v]
    except ImportError:
        pass
    if hasattr(go_graph, "edges"):
        edges = list(go_graph.edges())
        return [(e[0], e[1]) for e in edges if len(e) >= 2]
    if isinstance(go_graph, dict):
        if "edges" in go_graph:
            edges = list(go_graph["edges"])
            return [(e[0], e[1]) for e in edges if len(e) >= 2]
        if "edge_index" in go_graph and torch.is_tensor(go_graph["edge_index"]):
            edge_index = go_graph["edge_index"]
            if edge_index.dim() == 2 and edge_index.size(0) == 2:
                return edge_index.t().tolist()
    if torch.is_tensor(go_graph):
        if go_graph.dim() == 2 and go_graph.size(0) == 2:
            return go_graph.t().tolist()
        if go_graph.dim() == 2 and go_graph.size(0) == go_graph.size(1):
            return (go_graph > 0).nonzero(as_tuple=False).tolist()
    if isinstance(go_graph, (list, tuple)):
        out = []
        for e in go_graph:
            if len(e) >= 2:
                out.append((e[0], e[1]))
        return out
    raise ValueError("Unsupported go_graph format.")


def build_direct_parent_mask(num_labels, edges, edge_direction="child_to_parent", id2idx=None):
    mask = torch.zeros(num_labels, num_labels, dtype=torch.float32)
    parents_of = defaultdict(list)
    children_of = defaultdict(list)

    for u, v in edges:
        if id2idx is not None:
            u = id2idx[u]
            v = id2idx[v]
        u, v = int(u), int(v)
        if edge_direction == "parent_to_child":
            parent, child = u, v
        else:
            parent, child = v, u
        if 0 <= child < num_labels and 0 <= parent < num_labels:
            mask[child, parent] = 1.0
            parents_of[child].append(parent)
            children_of[parent].append(child)
    return mask, parents_of, children_of

def build_level_groups(num_labels, parents_of, children_of):
    indeg = {i: len(parents_of.get(i, [])) for i in range(num_labels)}
    depth = {i: 0 for i in range(num_labels) if indeg[i] == 0}
    q = deque([i for i in range(num_labels) if indeg[i] == 0])
    while q:
        p = q.popleft()
        for ch in children_of.get(p, []):
            nd = depth[p] + 1
            if ch not in depth or nd < depth[ch]:
                depth[ch] = nd
                q.append(ch)
    for i in range(num_labels):
        if i not in depth:
            depth[i] = 0

    levels = defaultdict(list)
    for idx, d in depth.items():
        levels[d].append(idx)

    level_groups = [torch.tensor(levels[d], dtype=torch.long) for d in sorted(levels.keys())]
    return level_groups


class TopDownHierarchicalHead(nn.Module):
    def __init__(
        self,
        num_labels,
        go_graph,
        protein_dim,
        label_dim,
        hidden_dim=None,
        edge_direction="child_to_parent",
        id2idx=None,
        beta_init=0.05,
    ):
        super().__init__()
        hidden_dim = hidden_dim or protein_dim
        edges = extract_edges(go_graph)
        direct_parent_mask, parents_of, children_of = build_direct_parent_mask(
            num_labels=num_labels,
            edges=edges,
            edge_direction=edge_direction,
            id2idx=id2idx,
        )
        self.register_buffer("direct_parent_mask", direct_parent_mask)
        self.register_buffer("parent_count", direct_parent_mask.sum(dim=1).clamp(min=1.0))
        self.level_groups = build_level_groups(num_labels, parents_of, children_of)
        self.update_mlp = nn.Sequential(
            nn.Linear(protein_dim + label_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )
        self.beta = nn.Parameter(torch.tensor(beta_init))
        self.child_gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, decoded_seq, label_features, temperature):
        B, D = decoded_seq.shape
        C = label_features.size(0)
        q = F.normalize(decoded_seq, p=2, dim=1)
        labels = F.normalize(label_features, p=2, dim=1)
        base_logits = torch.matmul(q, labels.t()) / temperature
        refined = base_logits.clone()
        for depth in range(1, len(self.level_groups)):
            child_idx = self.level_groups[depth].to(decoded_seq.device)
            if child_idx.numel() == 0:
                continue
            parent_ctx_all = torch.matmul(refined, self.direct_parent_mask.t())  # [B, C]
            parent_ctx_all = parent_ctx_all / self.parent_count.unsqueeze(0)
            child_ctx = parent_ctx_all[:, child_idx].unsqueeze(-1)  # [B, n_child, 1]
            protein_exp = decoded_seq.unsqueeze(1).expand(-1, child_idx.numel(), -1)  # [B, n_child, D]
            label_exp = label_features[child_idx].unsqueeze(0).expand(B, -1, -1)      # [B, n_child, D]
            child_in = torch.cat([protein_exp, label_exp, child_ctx], dim=-1)
            delta = self.update_mlp(child_in).squeeze(-1)  # [B, n_child]
            refined[:, child_idx] = (
                base_logits[:, child_idx]
                + torch.tanh(self.beta) * delta
                + torch.sigmoid(self.child_gate) * child_ctx.squeeze(-1)
            )
        return refined

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GCN, self).__init__()
        self.conv1 = dglnn.GraphConv(in_feats, hidden_feats, activation=F.relu)
        self.conv2 = dglnn.GraphConv(hidden_feats, out_feats)
    
    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = self.conv2(g, h)
        return h

class GlobalContextEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linearLayer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Dropout(config.DROPOUT),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(config.DROPOUT),
            nn.ReLU()
        )
    def forward(self, fixed_null_input):
        global_Context = self.linearLayer(fixed_null_input)
        return global_Context

class FAAttentionFusion(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128,dropout=0.1):
        super().__init__()
        self.attn_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, seq_features, struct_features, domain_features):
        feats = torch.stack(
            [seq_features, struct_features, domain_features],
            dim=1
        )  # [B, 3, D]
        scores = self.attn_mlp(feats)        # [B, 3, 1]
        weights = F.softmax(scores, dim=1)   # [B, 3, 1]
        fused = (weights * feats).sum(dim=1) # [B, D]

        return fused, weights


class ProteinQFormerDecoder(nn.Module):
    def __init__(self, seq_dim, label_dim, num_queries=8, num_heads=8, dropout=0.1):
        super(ProteinQFormerDecoder, self).__init__()
        self.num_queries = num_queries
        self.label_dim = label_dim
        self.query_tokens = nn.Parameter(torch.empty(num_queries, label_dim))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        self.query_modulator = nn.Sequential(
            nn.Linear(seq_dim, label_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(label_dim * 2, num_queries * label_dim)
        )
        nn.init.zeros_(self.query_modulator[-1].weight)
        nn.init.zeros_(self.query_modulator[-1].bias)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=label_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm_self = nn.LayerNorm(label_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=label_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm_cross = nn.LayerNorm(label_dim)
        self.ffn = nn.Sequential(
            nn.Linear(label_dim, label_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(label_dim * 4, label_dim),
            nn.Dropout(dropout)
        )
        self.norm_ffn = nn.LayerNorm(label_dim)

    def forward(self, seq_features, label_features):
        B = seq_features.size(0)
        base_query = self.query_tokens.unsqueeze(0).expand(B, -1, -1)
        modulation = self.query_modulator(seq_features).view(B, self.num_queries, self.label_dim)
        query = base_query + modulation  # [B, num_queries, label_dim]
        q_sa, _ = self.self_attn(query, query, query)
        query = self.norm_self(query + q_sa)
        kv = label_features.unsqueeze(0).expand(B, -1, -1)  # [B, N, label_dim]
        q_ca, attn_weights = self.cross_attn(query, kv, kv)
        query = self.norm_cross(query + q_ca)
        query = self.norm_ffn(query + self.ffn(query))
        output = query.mean(dim=1)
        return output, attn_weights


class GOFormer(nn.Module):
    def __init__(self, config, go_graph, pretrained_label_embeds=None):
        super(GOFormer, self).__init__()
        self.config = config
        self.go_graph = go_graph
        if pretrained_label_embeds is not None:
            num_labels, input_dim = pretrained_label_embeds.shape
            assert num_labels == config.NUM_LABELS, \
                f"预训练特征行数({num_labels})与配置的标签数({config.NUM_LABELS})不一致！"
            self.node_embed = nn.Embedding.from_pretrained(
                pretrained_label_embeds, 
                freeze=False 
            )
            if input_dim != config.HIDDEN_DIM:
                self.label_projector = nn.Sequential(
                    nn.Linear(input_dim, config.HIDDEN_DIM),
                    nn.LayerNorm(config.HIDDEN_DIM),
                    nn.GELU()
                )
            else:
                self.label_projector = nn.Identity()
        else:
            self.node_embed = nn.Embedding(config.NUM_LABELS, config.HIDDEN_DIM)
            self.label_projector = nn.Identity()
        self.seq_encoder = nn.Sequential(
            nn.Linear(config.ESM_DIM, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM)
        )
        self.struct_encoder = nn.Sequential(
            nn.Linear(config.STRUCT_DIM, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM)
        )
        self.globalContextEncoder = GlobalContextEncoder(config.null_input_dim, config.HIDDEN_DIM)
        self.feature_fusion = FAAttentionFusion(
            feature_dim=config.HIDDEN_DIM,
            hidden_dim=config.HIDDEN_DIM*3,
            dropout=config.DROPOUT
        )
        self.gcn = GCN(config.HIDDEN_DIM, config.HIDDEN_DIM * 2, config.HIDDEN_DIM)
        self.gcn_norm = nn.LayerNorm(config.HIDDEN_DIM)
        self.decoder = ProteinQFormerDecoder(
            seq_dim=config.HIDDEN_DIM,
            label_dim=config.HIDDEN_DIM,
            num_queries=config.NUM_QUERIES,
            num_heads=8,
            dropout=config.DROPOUT
        )
        self.dense_classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM + config.HIDDEN_DIM + config.HIDDEN_DIM + config.HIDDEN_DIM, config.HIDDEN_DIM*4),
            nn.LayerNorm(config.HIDDEN_DIM*4),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM*4, config.HIDDEN_DIM*4),
            nn.LayerNorm(config.HIDDEN_DIM*4),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM*4, config.NUM_LABELS)
        )
        self.temperature = nn.Parameter(torch.ones(1) * 0.07) 
        self.fusion_gate = nn.Parameter(torch.tensor(0.5))
        self.parent_graph_head = TopDownHierarchicalHead(
            num_labels=config.NUM_LABELS,
            go_graph=go_graph,
            protein_dim=config.HIDDEN_DIM,
            label_dim=config.HIDDEN_DIM,
            hidden_dim=config.HIDDEN_DIM,
            edge_direction=getattr(config, "GO_EDGE_DIRECTION", "child_to_parent"),
            id2idx=getattr(config, "GO_ID2IDX", None),
            beta_init=getattr(config, "TOPDOWN_BETA_INIT", 0.1),
        )
    def forward(self, esm_embeddings, fixed_null_input=None, struct=None, return_features=False, return_attention=False):
        seq_features = self.seq_encoder(esm_embeddings) # [B, Dim]
        global_Context = self.globalContextEncoder(fixed_null_input)
        struct = struct.squeeze(1)
        struct_features = self.struct_encoder(struct)
        fused_protein_features, weight = self.feature_fusion(seq_features, struct_features, global_Context)
        raw_embeds = self.node_embed.weight 
        projected_embeds = self.label_projector(raw_embeds) # [N, HIDDEN_DIM]
        label_features = self.gcn_norm(projected_embeds)
        decoded_seq, attn_weights = self.decoder(fused_protein_features, label_features)
        logits_dense = self.dense_classifier(torch.cat((decoded_seq,seq_features,global_Context,struct_features),1))
        logits_graph = self.parent_graph_head(
            decoded_seq=decoded_seq,
            label_features=label_features,
            temperature=self.temperature
        )
        # Fusion
        gate = torch.sigmoid(self.fusion_gate)
        logits = gate * logits_dense + (1 - gate) * logits_graph
        return logits, decoded_seq, label_features, logits_dense, logits_graph,  fused_protein_features

