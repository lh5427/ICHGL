import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_conv import GraphConvLayer
from .trans import TRANS


class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()
        self.gamma = 1e-10

    def forward(self, p_score, n_score):
        loss = -torch.log(self.gamma + torch.sigmoid(p_score - n_score))
        return loss.mean()



# 个性化迁移模块
class TransferLoss(nn.Module):

    def __init__(self):
        super(TransferLoss, self).__init__()
    def forward(self, final_emb, compare_emb_list):
        """
         final_emb: [N, D] compare_emb_list: list of [N, D]
        """
        loss_sum = 0
        for emb in compare_emb_list:
            cos = F.cosine_similarity(final_emb, emb, dim=1) # [N]
            loss = (1 - cos).mean()
            loss_sum += loss
        return loss_sum



class ICHGL(nn.Module):
    '''Multi-Grained Graph Learning for Multi-Behavior Recommendation'''

    def __init__(self, data, emb_dim, gnn_layers, tide_layers, lambda_mi, mi_temp, lambda_con):
        super(ICHGL, self).__init__()
        self.edge_dict = data['edge_dict']

        self.n_users = data['n_users']
        self.n_items = data['n_items']
        self.emb_dim = emb_dim

        self.gnn_layers = gnn_layers[0] if isinstance(gnn_layers, list) else gnn_layers
        self.tide_layers = tide_layers[0] if isinstance(tide_layers, list) else tide_layers
        self.lambda_con = lambda_con[0] if isinstance(lambda_con, list) else lambda_con
        # ===========================================================

        self.bsg_types = data['bsg_types']
        self.tcb_types = data['tcb_types']
        self.trbg_types = self.tcb_types

        self.bsg1_types = [f"{bt}_tide" for bt in self.bsg_types if bt != 'buy']

        self.bsg_only_types = ['view_only', 'cart_only']
        if 'collect' in self.bsg_types:
            self.bsg_only_types.append('collect_only')
        self.bsg_only_types.append('buy_only')

        self.bsg_overlap_types = data['bsg_overlap_types']

        self.ubg_minus_bsg_only_types = data.get('ubg_minus_bsg_only_types', [])
        self.total_behaviors =(['ubg'] + self.bsg_types + self.trbg_types + self.bsg1_types
            + self.bsg_only_types + self.bsg_overlap_types + self.ubg_minus_bsg_only_types)


        self.buy_minus_buy_only_types = data.get('buy_minus_buy_only_types', [])
        self.total_behaviors = (['ubg'] + self.bsg_types + self.trbg_types + self.bsg1_types
                                + self.bsg_only_types + self.bsg_overlap_types + self.buy_minus_buy_only_types)


        self.trans = TRANS(emb_dim, self.bsg_types, self.tcb_types)
        self.bpr_loss = BPRLoss()

        self.transfer_loss = TransferLoss()

        self.user_embedding = nn.Embedding(self.n_users + 1, emb_dim, padding_idx=0)  # index 0 is padding
        self.item_embedding = nn.Embedding(self.n_items + 1, emb_dim, padding_idx=0)  # index 0 is padding

        self.convs = nn.ModuleDict()
        for behavior_type in self.total_behaviors:
            if behavior_type in self.bsg_types and behavior_type != 'buy':
                self.convs[behavior_type] = nn.ModuleList(
                    [GraphConvLayer(emb_dim, emb_dim, 'tide') for _ in range(self.tide_layers)])
            else:
                self.convs[behavior_type] = nn.ModuleList(
                    [GraphConvLayer(emb_dim, emb_dim, 'gcn') for _ in range(self.gnn_layers)])

        self.lambda_mi = lambda_mi
        self.mi_temp = 0.2
        self.lambda_con = self.lambda_con
        self.reset_parameters()



    def reset_parameters(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        self.trans.reset_parameters()


    def info_nce_loss(self, anchor, target, temperature=0.2, eps=1e-8, negative_samples=1024):

        anchor = F.normalize(anchor, dim=1)
        target = F.normalize(target, dim=1)
        N = anchor.size(0)

        if N > negative_samples:
            sampled_indices = torch.randperm(N, device=anchor.device)[:negative_samples]
            target_sampled = target[sampled_indices]  # [K, d]
            pos_sim = torch.sum(anchor * target, dim=1, keepdim=True) / (temperature + eps)  # [N, 1]
            neg_sim = torch.matmul(anchor, target_sampled.t()) / (temperature + eps)  # [N, K]
            mask_self = torch.isin(sampled_indices, torch.arange(N, device=anchor.device))
            if mask_self.any():
                neg_sim[:, mask_self] = -float("inf")

            all_sim = torch.cat([pos_sim, neg_sim], dim=1)  # [N, 1 + K]
            loss = -pos_sim + torch.logsumexp(all_sim, dim=1, keepdim=True)
            return loss.mean()
        else:
            sim = torch.matmul(anchor, target.t()) / (temperature + eps)  # [N, N]
            pos = torch.diag(sim).unsqueeze(1)  # [N, 1]
            loss = -pos + torch.logsumexp(sim, dim=1, keepdim=True)
            return loss.mean()


    def propagate(self, x, edge_index, behavior_type, target_emb=None, aux_target_emb=None):
        result = [x]
        mi_loss_accum = 0.0
        for i, conv in enumerate(self.convs[behavior_type]):

            pre_conv_x = x

            if behavior_type in self.bsg_types and behavior_type != 'buy':
                x = conv(x, edge_index, aux_target_emb=aux_target_emb)
            else:
                x = conv(x, edge_index, target_emb=target_emb)

            x = F.normalize(x, dim=1)
            result.append(x / (i + 1))

            if target_emb is not None:
                mi_loss_layer = self.info_nce_loss(target_emb, pre_conv_x, temperature=self.mi_temp)
                mi_loss_accum = mi_loss_accum + mi_loss_layer

        result = torch.stack(result, dim=1)
        x = result.sum(dim=1)

        if target_emb is not None:
            return x, mi_loss_accum
        else:
            return x


    def forward(self):
        edge_dict = self.edge_dict
        emb_dict = dict()
        init_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        ubg_emb = self.propagate(init_emb, edge_dict['ubg'], 'ubg')
        emb_dict['ubg'] = ubg_emb

        if 'buy' in self.bsg_types:
            buy_emb = self.propagate(emb_dict['ubg'], edge_dict['buy'], 'buy')
            emb_dict['buy'] = buy_emb


        for behavior_type in self.bsg_types:
            if behavior_type != 'buy':
                previous_emb = emb_dict['ubg']
                aux_target_emb = emb_dict['buy']
                bsg_emb = self.propagate(previous_emb, edge_dict[behavior_type], behavior_type, aux_target_emb = aux_target_emb)
                emb_dict[behavior_type] = bsg_emb



        total_mi_loss = 0.0

        for behavior_type in self.tcb_types:
            previous_behavior = behavior_type.split('_')[0]
            previous_emb = emb_dict[previous_behavior]
            target_emb = emb_dict['buy']
            tcb_emb, mi_loss = self.propagate(previous_emb, edge_dict[behavior_type], behavior_type, target_emb=target_emb)     # , aux_target_emb=emb_dict['buy'])
            emb_dict[behavior_type] = tcb_emb



            if isinstance(total_mi_loss, float):
                total_mi_loss = mi_loss
            else:
                total_mi_loss = total_mi_loss + mi_loss


        final_emb = self.trans(emb_dict)
        emb_dict['final'] = final_emb
        return emb_dict, total_mi_loss


    def calc_transfer_loss(self, emb_dict):

        final = emb_dict['final']
        compare_list = []

        # final vs UBG
        compare_list.append(emb_dict['ubg'])
        # final vs BSG behaviors
        for b in self.bsg_types:
            compare_list.append(emb_dict[b])

        transfer_loss = self.transfer_loss(final, compare_list)
        return transfer_loss


    def loss(self, users, pos_idx, neg_idx):

        emb_dict, mi_loss = self.forward()
        user_emb, item_emb = torch.split(emb_dict['final'], [self.n_users + 1, self.n_items + 1], dim=0)
        p_score = (user_emb[users] * item_emb[pos_idx]).sum(dim=1)
        n_score = (user_emb[users] * item_emb[neg_idx]).sum(dim=1)
        bpr_loss = self.bpr_loss(p_score, n_score)
        transfer = self.calc_transfer_loss(emb_dict)
        total_loss = bpr_loss + self.lambda_mi * mi_loss + self.lambda_con * transfer
        return total_loss


    def predict(self, users):

        emb_dict, _ = self.forward()
        final_embeddings = emb_dict['final']
        final_user_emb, final_item_emb = torch.split(final_embeddings, [self.n_users + 1, self.n_items + 1])
        user_emb = final_user_emb[users.long()]
        scores = torch.matmul(user_emb, final_item_emb.transpose(0, 1))
        return scores

