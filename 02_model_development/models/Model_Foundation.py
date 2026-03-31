"""
Model Architecture — HGSurv (Hybrid Graph-Transformer Survival Model)
=====================================================================
A dual-branch WSI-level survival prediction model that fuses local graph
learning (dynamic KNN + GCN) with global Transformer context, followed
by interpretable attention pooling and a Cox regression head.

Module inventory:
    - GatedLinearUnit          : GLU-based nonlinear projection
    - GraphAttentionLayer      : GAT layer with efficient broadcasting attention
    - GraphConvolution         : GCN layer with residual connection
    - SelfAttention            : Standard Transformer encoder block
    - AttnPooling              : Attention-based MIL pooling
    - HGSurv                   : Main model (interface unchanged)
    - cox_loss                 : Negative partial log-likelihood (Cox PH)
    - AdvancedLoss             : Cox loss + L1 embedding regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# =========================================================================
# 1. Building Blocks
# =========================================================================

class GatedLinearUnit(nn.Module):
    """
    Gated Linear Unit (GLU).
    Provides a richer nonlinear gating mechanism compared to ReLU,
    commonly used in advanced NLP and GNN architectures.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        value, gate = x.chunk(2, dim=-1)
        return value * torch.sigmoid(gate)


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Network (GAT) layer with efficient additive attention.
    Uses separate projections for source and target nodes, then combines
    via broadcasting to avoid O(N^2) memory from explicit concatenation.
    """

    def __init__(self, in_features: int, out_features: int,
                 dropout: float = 0.3, alpha: float = 0.2):
        super().__init__()
        self.dropout = dropout
        self.W = nn.Linear(in_features, out_features, bias=False)

        # Separate attention parameters for source and target nodes
        self.a1 = nn.Parameter(torch.empty(out_features, 1))
        self.a2 = nn.Parameter(torch.empty(out_features, 1))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        """
        Args:
            x:   Node features [B, N, in_features]
            adj: Adjacency matrix [B, N, N]

        Returns:
            h_prime:                Updated node features [B, N, out_features]
            edge_attention_scores:  Attention weights [B, N, N]
        """
        h = self.W(x)  # [B, N, out_features]

        # Compute attention scores via additive decomposition (avoids N^2 concat)
        attn_for_self = torch.matmul(h, self.a1)    # [B, N, 1]
        attn_for_neighs = torch.matmul(h, self.a2)  # [B, N, 1]

        # Broadcasting: [B, N, 1] + [B, 1, N] -> [B, N, N]
        e = self.leakyrelu(attn_for_self + attn_for_neighs.transpose(1, 2))

        # Masked attention: set non-edges to -inf before softmax
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        edge_attention_scores = F.softmax(attention, dim=-1)
        attention_dropped = F.dropout(
            edge_attention_scores, self.dropout, training=self.training
        )

        h_prime = torch.matmul(attention_dropped, h)
        return F.elu(h_prime), edge_attention_scores


class GraphConvolution(nn.Module):
    """
    Graph Convolution layer with residual connection.
    Performs neighbourhood aggregation to capture local spatial interactions.
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.3):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   Node features [B, N, C]
            adj: Normalized adjacency matrix [B, N, N]
        """
        support = self.linear(x)
        output = torch.matmul(adj, support)  # Message passing / aggregation
        output = self.norm(output)
        return self.dropout(self.activation(output)) + x  # Residual connection


class SelfAttention(nn.Module):
    """
    Standard Transformer Encoder block (Pre-LN variant).
    Captures global context across all patches.
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.mha(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)


class AttnPooling(nn.Module):
    """
    Gated Attention Pooling for Multiple Instance Learning (MIL).
    Produces a slide-level embedding and interpretable per-patch
    attention weights indicating each patch's risk contribution.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Patch features [B, N, C]

        Returns:
            M: Aggregated slide embedding [B, C]
            a: Attention weights [B, N, 1]
        """
        a = self.attention(x)         # [B, N, 1]
        a = torch.softmax(a, dim=1)
        M = torch.sum(x * a, dim=1)   # [B, C]
        return M, a


# =========================================================================
# 2. Main Model: HGSurv
# =========================================================================

class HGSurv(nn.Module):
    """
    Hybrid Graph-Transformer Survival model.
    Dual-branch architecture that fuses:
        - Branch 1: Dynamic KNN graph + GCN (local spatial topology)
        - Branch 2: Transformer encoder (global context)
    followed by attention pooling and a log-hazard prediction head.
    """

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 512,
                 dropout: float = 0.25):
        super().__init__()

        # 1. Feature compression and encoding
        self.fc_start = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            GatedLinearUnit(hidden_dim),
            nn.Dropout(dropout),
        )

        # 2. Local graph branch (spatial topology)
        self.gcn_layers = nn.ModuleList([
            GraphConvolution(hidden_dim, hidden_dim, dropout),
            GraphConvolution(hidden_dim, hidden_dim, dropout),
        ])

        # 3. Global Transformer branch (sequential context)
        self.trans_layers = nn.ModuleList([
            SelfAttention(hidden_dim, num_heads=8, dropout=dropout),
            SelfAttention(hidden_dim, num_heads=8, dropout=dropout),
        ])

        # 4. Fusion and pooling
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.pool = AttnPooling(hidden_dim)

        # 5. Prediction head (outputs log-hazard)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def create_knn_graph(self, x: torch.Tensor, k: int = 8) -> torch.Tensor:
        """
        Dynamically construct a KNN graph in the feature space.

        For very large N (>5000 patches), consider pre-computing a static
        graph based on spatial coordinates to save GPU memory.

        Args:
            x: Node features [B, N, C]
            k: Number of nearest neighbours

        Returns:
            adj: Symmetrically normalized adjacency matrix [B, N, N]
        """
        B, N, C = x.shape

        # Pairwise distance in feature space
        dist = torch.cdist(x, x)  # [B, N, N]

        # Select K nearest neighbours (k+1 because self-loop is included)
        _, indices = torch.topk(dist, k=k + 1, largest=False, dim=-1)

        # Build binary adjacency matrix via advanced indexing
        adj = torch.zeros(B, N, N, device=x.device)
        batch_idx = torch.arange(B, device=x.device).view(-1, 1, 1)
        row_idx = torch.arange(N, device=x.device).view(1, -1, 1)
        adj[batch_idx, row_idx, indices] = 1.0

        # Symmetric normalization: D^{-0.5} A D^{-0.5}
        row_sum = adj.sum(dim=2)
        d_inv_sqrt = row_sum.pow(-0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat = torch.diag_embed(d_inv_sqrt)
        adj = d_mat @ adj @ d_mat

        return adj

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: WSI patch features [B, N, input_dim]

        Returns:
            logits:       Log-hazard prediction [B, 1]
            attn_weights: Per-patch attention weights [B, N, 1]
            slide_emb:    Slide-level embedding [B, hidden_dim]
        """
        h = self.fc_start(x)  # [B, N, hidden_dim]

        # --- Branch 1: Graph Learning (Local) ---
        adj = self.create_knn_graph(h, k=8)
        h_graph = h
        for gcn_layer in self.gcn_layers:
            h_graph = gcn_layer(h_graph, adj)

        # --- Branch 2: Transformer Learning (Global) ---
        h_trans = h
        for trans in self.trans_layers:
            h_trans = trans(h_trans)

        # --- Fusion ---
        h_combined = torch.cat([h_graph, h_trans], dim=-1)  # [B, N, hidden_dim*2]
        h_fused = self.fusion(h_combined)                    # [B, N, hidden_dim]

        # --- Pooling ---
        slide_emb, attn_weights = self.pool(h_fused)

        # --- Prediction ---
        logits = self.classifier(slide_emb)

        return logits, attn_weights, slide_emb


# =========================================================================
# 3. Loss Functions
# =========================================================================

def cox_loss(survtime, censor, hazard_pred, device):
    """
    Negative partial log-likelihood loss for the Cox Proportional Hazards model.

    Args:
        survtime:    Survival times   (B,)  — can be a Tensor or array-like.
        censor:      Event indicators (B,)  — 1 = event occurred, 0 = censored.
        hazard_pred: Predicted log-hazard (B, 1) from the model.
        device:      Target torch device.

    Returns:
        Scalar loss tensor.
    """
    current_batch_len = len(survtime)

    # Build the risk-set indicator matrix R[i, j] = 1 if time_j >= time_i
    # Vectorised version avoids the O(N^2) Python loop
    if isinstance(survtime, torch.Tensor):
        t = survtime.detach().cpu().float()
    else:
        t = torch.tensor(survtime, dtype=torch.float32)

    # Broadcasting: t_j (row) >= t_i (col) -> [B, B]
    R_mat = (t.unsqueeze(0) >= t.unsqueeze(1)).float().to(device)

    theta = hazard_pred.reshape(-1)
    exp_theta = torch.exp(theta)

    # Numerical stability: add epsilon inside the log
    loss = -torch.mean(
        (theta - torch.log(torch.sum(exp_theta * R_mat, dim=1) + 1e-8)) * censor
    )
    return loss


class AdvancedLoss(nn.Module):
    """
    Combined survival loss:
        total = Cox_loss + alpha * L1_regularization(slide_embedding)

    The L1 penalty encourages embedding sparsity, which helps prevent
    overfitting by forcing the model to retain only the most informative
    feature dimensions.
    """

    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha  # Regularization coefficient

    def forward(self, logits, time, event, slide_emb):
        """
        Args:
            logits:    Predicted log-hazard [B, 1]
            time:      Survival times [B]
            event:     Event indicators [B]
            slide_emb: Slide-level embedding [B, C]
        """
        # Primary loss: Cox negative partial log-likelihood
        loss_main = cox_loss(time, event, logits, logits.device)

        # Auxiliary loss: L1 regularization on the slide embedding
        loss_reg = torch.mean(torch.abs(slide_emb))

        return loss_main + self.alpha * loss_reg


# =========================================================================
# 4. Quick Sanity Check (only runs when executed directly)
# =========================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HGSurv(input_dim=1024).to(device)
    criterion = AdvancedLoss(alpha=1e-4)

    # Optimizer: AdamW with CosineAnnealing schedule
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=1e-6
    )

    # Smoke test with random input
    dummy_input = torch.randn(1, 100, 1024).to(device)
    logits, attn_weights, slide_emb = model(dummy_input)
    print(f"Model initialised on {device}.")
    print(f"  logits shape:       {logits.shape}")
    print(f"  attn_weights shape: {attn_weights.shape}")
    print(f"  slide_emb shape:    {slide_emb.shape}")
    print("Sanity check passed.")

