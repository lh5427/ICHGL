import os
import torch
import argparse  # 添加这行
from data import load_data
from model import ICHGL, Trainer
from parser import parse_args
from utils import set_seed, print_args
from loguru import logger



def run_experiment_with_params(args, gnn_layer, tide_layer, lambda_con_val, lr_val, weight_decay_val):
    """使用特定参数运行一次完整实验"""
    # 创建当前实验的参数副本
    current_args = argparse.Namespace(**vars(args))
    current_args.gnn_layers = gnn_layer
    current_args.tide_layers = tide_layer
    current_args.lambda_con = lambda_con_val
    current_args.lr = lr_val
    current_args.weight_decay = weight_decay_val


    if hasattr(current_args, "seeds") and current_args.seeds is not None:
        seed_list = current_args.seeds
    else:
        seed_list = [current_args.seed if current_args.seed is not None else 42]

    all_results = {}  # key = seed, value = (HR, NDCG)

    for seed in seed_list:
        logger.info(
            f"\n========== Running Experiment with gnn_layers={gnn_layer}, tide_layers={tide_layer}, lambda_con={lambda_con_val}, Seed {seed} ==========")
        set_seed(seed)


        data = load_data(current_args.data_dir, current_args.dataset, current_args.device, current_args.batch_size)

        model = ICHGL(
            data,
            current_args.emb_dim,
            current_args.gnn_layers,
            current_args.tide_layers,
            lambda_mi=current_args.lambda_mi,
            mi_temp=current_args.mi_temp,
            lambda_con=current_args.lambda_con
        ).to(current_args.device)

        trainer = Trainer(model, data, current_args)

        if current_args.load_checkpoint:
            logger.info(
                f"Load checkpoint from {os.path.join(current_args.checkpoint_dir, current_args.dataset, 'model.pt')}")
            model.load_state_dict(torch.load(
                os.path.join(current_args.checkpoint_dir, current_args.dataset, 'model.pt'),
                map_location=current_args.device
            ))
        else:
            logger.info("Start training the model")
            trainer.train_model()

        logger.info("Start evaluating the model")
        hr, ndcg = trainer.evaluate()
        logger.info(f"[Seed {seed}] HR@{current_args.topk}: {hr:.4f}, NDCG@{current_args.topk}: {ndcg:.4f}")

        # 保存当前 seed 的结果
        all_results[seed] = (hr, ndcg)




    avg_hr = sum(hr for hr, _ in all_results.values()) / len(all_results)
    avg_ndcg = sum(ndcg for _, ndcg in all_results.values()) / len(all_results)

    logger.info(
        f"\n========== Summary for Parameter Set (gnn={gnn_layer}, tide={tide_layer}, lambda_con={lambda_con_val}) ==========")
    for seed, (hr, ndcg) in all_results.items():
        logger.info(f"Seed {seed}: HR@{current_args.topk}={hr:.4f}, NDCG@{current_args.topk}={ndcg:.4f}")
    logger.info(f"Average: HR@{current_args.topk}={avg_hr:.4f}, NDCG@{current_args.topk}={avg_ndcg:.4f}")

    return avg_hr, avg_ndcg

# 在 main 函数之前添加一个新的函数，并替换下面的 main 函数为以下内容。
def main(args):
    print_args(args)


    gnn_layers_list = args.gnn_layers if isinstance(args.gnn_layers, list) else [args.gnn_layers]
    tide_layers_list = args.tide_layers if isinstance(args.tide_layers, list) else [args.tide_layers]
    lambda_con_list = args.lambda_con if isinstance(args.lambda_con, list) else [args.lambda_con]
    lr_list = args.lr if isinstance(args.lr, list) else [args.lr]
    weight_decay_list = args.weight_decay if isinstance(args.weight_decay, list) else [args.weight_decay]

    all_experiments = []


    for gnn_layer in gnn_layers_list:
        for tide_layer in tide_layers_list:
            for lambda_con_val in lambda_con_list:
                for lr_val in lr_list:
                    for weight_decay_val in weight_decay_list:
                        avg_hr, avg_ndcg = run_experiment_with_params(args, gnn_layer, tide_layer, lambda_con_val,
                                                                      lr_val, weight_decay_val)

                        all_experiments.append({
                            'params': (gnn_layer, tide_layer, lambda_con_val, lr_val, weight_decay_val),
                            'results': (avg_hr, avg_ndcg)
                        })



    logger.info("\n========== Final Summary for All Experiments ==========")
    for exp in all_experiments:
        params = exp['params']
        avg_hr, avg_ndcg = exp['results']
        logger.info(f"Params(gnn={params[0]}, tide={params[1]}, lambda_con={params[2]}): "
                    f"Avg HR@{args.topk}={avg_hr:.4f}, Avg NDCG@{args.topk}={avg_ndcg:.4f}")
    # ===========================================================


if __name__ == '__main__':
    args = parse_args()

    main(args)