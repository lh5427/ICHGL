import torch
import torch.nn as nn

class TRANS(nn.Module):
    """
    Multi-grained embedding aggregator (REPLACED by GAT-style two-level transfer fusion).
    """
    def __init__(self, emb_dim, bsg_types, tcb_types):
        super(TRANS, self).__init__()
        self.emb_dim = emb_dim
        self.bsg_types = bsg_types          # 第一层用到的辅助行为（view, cart）
        self.tcb_types = tcb_types        # 第二层用到的 TCB 行为
        concat_dim = emb_dim * 2
        ### 修改1：Linear 层变为 D→D（不再是 2D→D）
        self.W1 = nn.Linear(emb_dim, emb_dim, bias=False)
        # 消融构建辅助意图子图1.注释下面。
        self.W2 = nn.Linear(emb_dim, emb_dim, bias=False)
        # 消融构建辅助意图子图1.注释下面。
        self.alpha = 1  # (1-α)*buy + α*aux
        # 参数初始化
        nn.init.xavier_uniform_(self.W1.weight)
        # 消融构建辅助意图子图2.注释下面。
        nn.init.xavier_uniform_(self.W2.weight)
        # 消融构建辅助意图子图2.注释下面。

    def reset_parameters(self):
        # 保留接口（不会被外部调用）
        pass

    def forward(self, emb_dict):
        # ---------- 第一层: BSG 辅助行为 → BUY ----------
        buy_emb = emb_dict["buy"]                 # [N, D]
        aux_list = [emb_dict[b] for b in self.bsg_types]
        aux_emb = torch.stack(aux_list, dim=1)    # [N, B1, D]
        ### 修改2：分别线性映射
        buy_h = self.W1(buy_emb)                  # [N, D]
        aux_h = self.W1(aux_emb)                  # [N, B1, D]
        # buy_h: [N, D] → [N, 1, D] for broadcasting
        scores = (aux_h * buy_h.unsqueeze(1)).sum(dim=-1)   # [N, B1]
        att1 = torch.softmax(scores, dim=1)                 # [N, B1]
        # Σ att1_i * aux_i
        weighted_aux = (att1.unsqueeze(-1) * aux_emb).sum(dim=1)
        # 第一层融合结果
        fused_level1 = (1 - self.alpha) * buy_emb + self.alpha * weighted_aux

        # 消融构建辅助意图子图3.注释下面。
        # ---------- 第二层: TCB → 第一层融合结果 ----------
        # 将 tcb_types 和 ubg_minus_bsg_only_types 合并处理
        combined_types = self.tcb_types     # + self.tib_types      # + self.ubg_minus_bsg_only_types
        tcb_list = [emb_dict[b] for b in combined_types]
        tcb_emb = torch.stack(tcb_list, dim=1)  # [N, B2, D]

        ### 修改3：dot attention
        fused_h = self.W2(fused_level1)         # [N, D]
        tcb_h = self.W2(tcb_emb)                # [N, B2, D]
        scores2 = (tcb_h * fused_h.unsqueeze(1)).sum(dim=-1)   # [N, B2]
        att2 = torch.softmax(scores2, dim=1)                   # [N, B2]
        weighted_tcb = (att2.unsqueeze(-1) * tcb_emb).sum(dim=1)
        updated_emb = (1 - self.alpha) * fused_level1 + self.alpha * weighted_tcb

        return updated_emb