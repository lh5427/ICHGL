import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='ICHGL Settings')
    parser.add_argument('--dataset', type=str, default='taobao', help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='/data/lh24/ICHGL-main/data', help='Directory containing the data')
    parser.add_argument('--checkpoint_dir', type=str, default='/data/lh24/ICHGL-main/checkpoint', help='Directory of model checkpoint')
    parser.add_argument('--load_checkpoint', action='store_true', help='Load model checkpoint')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for target data')

    parser.add_argument('--lr', type=float, nargs='+', default=[1e-4],
                        help='List of lr')
    parser.add_argument('--weight_decay', type=float, nargs='+', default=[1e-3],
                        help='List of weight_decay')

    parser.add_argument('--gnn_layers', type=int, nargs='+', default=[1],
                        help='List of gnn layers')
    parser.add_argument('--tide_layers', type=int, nargs='+', default=[4],
                        help='List of tide layers')

    parser.add_argument('--lambda_mi', type=float, default=0.1, help='Mutual information loss weight')
    parser.add_argument('--mi_temp', type=float, default=0.2, help='Temperature for mutual information loss')

    parser.add_argument('--lambda_con', type=float, nargs='+', default=[0.1],
                        help='contrast loss weight')
    parser.add_argument('--emb_dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--seed', type=int, default=118, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='Training device')
    parser.add_argument('--topk', type=int, default=10, help='Top-k items')
    return parser.parse_args()
