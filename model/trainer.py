import torch
import numpy as np
import os

from loguru import logger
from tqdm import tqdm
from .metrics import ndcg, hit, recall   # 添加recall导入

import os
import torch


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


torch.backends.cudnn.benchmark = True

# 添加异常处理以捕获CUDA错误
try:
    pass
except RuntimeError as e:
    if "CUDA error" in str(e):
        print(f"CUDA Error: {e}")
        # 清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        raise e


class Trainer:
    def __init__(self, model, data, args):
        self.model = model
        self.data = data
        self.args = args
        self.topk = args.topk

        lr = args.lr[0] if isinstance(args.lr, list) else args.lr
        weight_decay = args.weight_decay[0] if isinstance(args.weight_decay, list) else args.weight_decay
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        train_loader = self.data['train_loader']
        with tqdm(total=len(train_loader), desc='Training', unit='batch', leave=False) as pbar:
            for user_indices, pos_indices, neg_indices in train_loader:
                user_indices, pos_indices, neg_indices = user_indices.to(self.args.device), pos_indices.to(self.args.device), neg_indices.to(self.args.device)

                self.optimizer.zero_grad()

                # 使用新的损失函数
                loss = self.model.loss(user_indices, pos_indices, neg_indices)

                loss.backward()
                self.optimizer.step()

                pbar.set_description(f'Batch Loss: {loss.item():.4f}')
                pbar.update()
                total_loss += loss.item()
        return total_loss / len(pbar)

    def train_model(self):
        num_epochs = self.args.num_epochs
        pbar = tqdm(range(num_epochs), desc='Epoch', unit='epoch', leave=False)
        for epoch in pbar:
            loss = self.train_epoch(epoch)
            pbar.set_description(f'Epoch {epoch+1} total loss: {loss:.4f}')
            pbar.update()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save trained model
        checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.args.dataset)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save(self.model.state_dict(), f'{checkpoint_dir}/model.pt')

    def evaluate(self):
        device = self.args.device

        self.model.eval()
        topk_list = []
        user_list = []
        with torch.no_grad():
            for user_indices, pos_indices, neg_indices in tqdm(self.data['test_loader'], desc='Test', leave=False):
                user_indices, pos_indices, neg_indices = user_indices.to(device), pos_indices.to(device), neg_indices.to(device)
                scores = self.model.predict(user_indices)

                for user_idx in range(user_indices.size(0)):
                    user = user_indices[user_idx].item()
                    train_items = self.data['train_gt'].get(str(user), [])
                    scores[user_idx, train_items] = -np.inf


                _, topk_indices = torch.topk(scores, max(50, self.topk), dim=1)


                for idx, user in enumerate(user_indices):
                    user_id = user.item()
                    user_list.append(user_id)  # ✅ 修复点：存入用户ID
                    gt_items = np.array(self.data['test_gt'][str(user.item())])
                    topk_items = topk_indices[idx].cpu().numpy()
                    mask = np.isin(topk_items, gt_items)
                    topk_list.append(mask)

        topk_list = np.vstack(topk_list)

        test_gt_length_for_users = np.array([len(self.data['test_gt'][str(user)]) for user in user_list])

        assert topk_list.shape[0] == test_gt_length_for_users.shape[0], \
            f"Mismatch: topk_list={topk_list.shape[0]}, test_gt_length={test_gt_length_for_users.shape[0]}"


        hr_res = hit(topk_list, test_gt_length_for_users).mean(axis=0)
        ndcg_res = ndcg(topk_list, test_gt_length_for_users).mean(axis=0)
        recall_res = recall(topk_list, test_gt_length_for_users).mean(axis=0)  # 添加recall计算


        k_values = [10, 50]
        results = {}
        print("\n" + "=" * 50)
        print("Final Evaluation Results:")
        print("=" * 50)
        for k in k_values:
            k_index = k - 1
            hr_k = hr_res[k_index]
            ndcg_k = ndcg_res[k_index]
            recall_k = recall_res[k_index]

            results[f"HR@{k}"] = hr_k
            results[f"NDCG@{k}"] = ndcg_k
            results[f"Recall@{k}"] = recall_k

            print(f"HR@{k}: {hr_k:.4f}")
            print(f"NDCG@{k}: {ndcg_k:.4f}")
            print(f"Recall@{k}: {recall_k:.4f}")
        print("=" * 50)


        hr_topk = hr_res[self.topk - 1] if self.topk <= len(hr_res) else hr_res[-1]
        ndcg_topk = ndcg_res[self.topk - 1] if self.topk <= len(ndcg_res) else ndcg_res[-1]


        return hr_topk, ndcg_topk




# import torch
# import numpy as np
# import os
#
# from loguru import logger
# from tqdm import tqdm
# from .metrics import ndcg, hit, recall
#
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
#
# torch.backends.cudnn.benchmark = True
#
#
# class Trainer:
#     def __init__(self, model, data, args):
#         self.model = model
#         self.data = data
#         self.args = args
#         self.topk = args.topk
#
#         lr = args.lr[0] if isinstance(args.lr, list) else args.lr
#         weight_decay = args.weight_decay[0] if isinstance(args.weight_decay, list) else args.weight_decay
#
#         self.optimizer = torch.optim.Adam(
#             self.model.parameters(),
#             lr=lr,
#             weight_decay=weight_decay
#         )
#
#     def train_epoch(self, epoch):
#         self.model.train()
#         total_loss = 0
#         train_loader = self.data['train_loader']
#
#         with tqdm(total=len(train_loader), desc='Training', unit='batch', leave=False) as pbar:
#             for user_indices, pos_indices, neg_indices in train_loader:
#                 user_indices = user_indices.to(self.args.device)
#                 pos_indices = pos_indices.to(self.args.device)
#                 neg_indices = neg_indices.to(self.args.device)
#
#                 self.optimizer.zero_grad()
#                 loss = self.model.loss(user_indices, pos_indices, neg_indices)
#                 loss.backward()
#                 self.optimizer.step()
#
#                 total_loss += loss.item()
#                 pbar.set_description(f'Batch Loss: {loss.item():.4f}')
#                 pbar.update()
#
#         return total_loss / len(train_loader)
#
#     def train_model(self):
#         num_epochs = self.args.num_epochs
#
#         for epoch in tqdm(range(num_epochs), desc='Epoch', unit='epoch'):
#             loss = self.train_epoch(epoch)
#
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
#
#         # ================== 关键修改：保存 model + data ==================
#         checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.args.dataset)
#         os.makedirs(checkpoint_dir, exist_ok=True)
#
#         checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
#
#         torch.save(
#             {
#                 "model_state_dict": self.model.state_dict(),
#                 "data": self.data
#             },
#             checkpoint_path
#         )
#
#         print(f"Checkpoint saved to {checkpoint_path}")
#
#     def evaluate(self):
#         device = self.args.device
#         self.model.eval()
#
#         topk_list = []
#         user_list = []
#
#         with torch.no_grad():
#             for user_indices, pos_indices, neg_indices in tqdm(
#                 self.data['test_loader'], desc='Test', leave=False
#             ):
#                 user_indices = user_indices.to(device)
#                 scores = self.model.predict(user_indices)
#
#                 for i, user in enumerate(user_indices):
#                     train_items = self.data['train_gt'].get(str(user.item()), [])
#                     scores[i, train_items] = -np.inf
#
#                 _, topk_indices = torch.topk(scores, max(50, self.topk), dim=1)
#
#                 for idx, user in enumerate(user_indices):
#                     user_id = user.item()
#                     user_list.append(user_id)
#
#                     gt_items = np.array(self.data['test_gt'][str(user_id)])
#                     topk_items = topk_indices[idx].cpu().numpy()
#                     mask = np.isin(topk_items, gt_items)
#                     topk_list.append(mask)
#
#         topk_list = np.vstack(topk_list)
#         gt_lens = np.array([len(self.data['test_gt'][str(u)]) for u in user_list])
#
#         hr_res = hit(topk_list, gt_lens).mean(axis=0)
#         ndcg_res = ndcg(topk_list, gt_lens).mean(axis=0)
#         recall_res = recall(topk_list, gt_lens).mean(axis=0)
#
#         print("\nFinal Evaluation Results")
#         for k in [10, 50]:
#             print(f"HR@{k}: {hr_res[k-1]:.4f}")
#             print(f"NDCG@{k}: {ndcg_res[k-1]:.4f}")
#             print(f"Recall@{k}: {recall_res[k-1]:.4f}")
#
#         return hr_res[self.topk - 1], ndcg_res[self.topk - 1]
