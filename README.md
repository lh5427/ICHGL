# ICHGL
This is the official code for **ICHGL** (Intent-Consistent Hierarchical Graph Learning with Cross-Behavior Transfer for Multi-Behavior Recommendation).




## Datasets
The statistics of datasets used in ICHGL are summarized as follows.  
| Dataset | Users  | Items  | Views       | Collects        | Carts         | Buys   |
|---------|--------:|--------:|-------------:|-----------------:|---------------:|--------:|
| Taobao  | 15,449 | 11,953 | 873,954 | -          | 195,476   | 92,180 |
| Beibei  | 21,716 | 7,977 | 2,412,586 | -         | 642,622   | 282,860|
| Tenrec  | 27,947 | 15,439 | 1,489,997 | 13,947   | 1,914     | 1,307  |

<!--<img src="./assets/data_statistics.png" width="500px" height="200px" title="data statistics"/>-->



## Usage
### Train our model from scratch
You can train the model with the best hyperparameters for each dataset by typing the following command in your terminal:

#### Train ICHGL in the `Taobao` dataset
```python
python main.py --dataset taobao \
               --lr 1e-4 \
               --weight_decay 1e-10 \
               --tide_layers 1 \
               --gnn_layers 2 \
               --mi_temp 0.5 \
               --emb_dim 64 \
               --num_epochs 100 \
               --batch_size 1024  \
               --lambda_con 0.3
```

#### Train ICHGL in the `Beibei` dataset
```python
python main.py --dataset beibei \
               --lr 1e-4 \
               --weight_decay 1e-10 \
               --tide_layers 4 \
               --gnn_layers 1 \
               --mi_temp 0.5 \
               --emb_dim 64 \
               --num_epochs 100 \
               --batch_size 1024 \
               --lambda_con 0.1
```

#### Train ICHGL in the `Tenrec` dataset
```python
python main.py --dataset tenrec \
               --lr 1e-4 \
               --weight_decay 1e-10 \
               --tide_layers 2 \
               --gnn_layers 1 \
               --mi_temp 0.5 \
               --emb_dim 64 \
               --num_epochs 100 \
               --batch_size 1024 \
               --lambda_con 0.001
```



### Validated hyperparameters of MuLe
We provide the validated hyperparameters of MuLe for each dataset to ensure reproducibility.

|Dataset| $\eta$ | $\lambda$ | $L_{\texttt{tide}}$ | $L_{\texttt{light}}$ | $d$ | $T$ | $B$
|-------|--------|-----------|---------------------|----------------------|-------|-------|-------|
|Taobao| 1e-4   | 0.3       | 1                   | 2                    | 64 | 100 | 1024 |
|Beibei| 1e-4   | 0.1       | 4                   | 1                    | 64 | 100 | 1024 |
|Tenrec| 1e-4   | 0.001     | 2                   | 1                    | 64 | 100 | 1024 |

**Description of each hyperparameter**
* $\eta$: learning rate of the Adam optimizer (`--lr`)
* $\lambda$: transfer consistency loss coefficient (`--lambda_con`)
* $L_{\texttt{tda}}$: number of TDA's layers (`--tide_layers`)
* $L_{\texttt{light}}$: number of LightGCN's layers (`--gnn_layers`)
* $d$: embedding dimension (`--emb_dim`)
* $T$: number of epochs (`--num_epochs`)
* $B$: batch size for target data (`--batch_size`)


