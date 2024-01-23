# Few-shot Link Prediction on Hyper-relational Facts

In this work, we attempt to automatically infer a missing entity in an n-ary fact about a particular relation given only limited instances. For instance, given the fact the "George V held the position of monarch of the Irish Free State", the algorithm proposed in this paper tries to automatically infer that "Steve Jobs held the position of chief executive officer of Pixar".

## Method illustration

![image](https://github.com/JiyaoWei/FLEN/assets/43932741/076ba6fb-8546-4ce9-a6d8-f9a781668370)

The main idea of this model is to capture relation meta information from limited support instances to predict a missing element in a query instance, as defined in ``models_metarh.py``. We approach it from two angles. Firstly, even though the few-shot relations have limited instances, the entities involved have abundant background data, which helps to generate fewshot relation representations. Secondly, inspired by the success of meta-learning methods (Munkhdalai and Yu 2017; Finn, Abbeel, and Levine 2017) in the few-shot learning ﬁeld, we can adjust relation representations with loss gradients of the support instances to obtain the shared knowledge therein, which we refer to as relation meta information.

## Steps to run the experiments

### Requirements
* ``Python 3.9.4 ``
* ``PyTorch 1.9.1``
* ``Numpy 1.19.2``
* ``Torch-scatter 2.0.8``

### The detail hyper-parameters
|   Parameter   |  F-Wikipeople  |  F-JF17K  |  F-WD50K  |
|  ----  | ----  | ----  | ----  |
|  Embedding dim  | 100 | 100 | 100 |
|  Batch size  | 1024 | 128 | 1024 |
|  Learning rate  | 1e-4 | 1e-4 | 1e-3 |
|  Relation weight  | 0.9 | 0.2 | 0.1 |
|  Neighbor number  | 30 | 10 | 20 |
|  Input dropout  | 0.5 | 0.5 | 0.5 |
|  Transformer layer  | 2 | 2 | 2 |
|  Transformer head  | 4 | 4 | 4 |
|  Transformer dropout  | 0.1 | 0.1 | 0.1 |

### Starting training and evaluation

* On F-JF17K
```
nohup python -u main_metarh.py --dataset F-JF17K --embed_dim 100 --seed 42 --embed_model HINGE --max_seq_length 9 --weight 0.2 --batch_size 128 --learning_rate 1e-4 --weight_decay 0 --dropout_i 0.5 --num_layers 2 --num_heads 4 --dropout_g 0.1 --max_neighbor 10 --qual_opn rotate --qual_n sum --qual_aggregate sum --epoch 300000 --eval_epoch 1000 --few_rel_aggregate mean --relation_learner gran --embedding_learner transe --shuffle_background 0 --early_stopping_patience 10 --use_neighbor 1 --use_pretrain 1 --use_in_train 1 --fine_tune 1 --few 5 --device 1 > log/jf17k_5_0.2_128_1e-4_0.5_2_4_0.1_10_rotate_sum_sum_300000_1000_mean_gran_transe_10_1_1_1_1_1 &
```
* On F-WD50K
```
nohup python -u main_metarh.py --dataset F-WD50K --embed_dim 100 --seed 42 --embed_model HINGE --max_seq_length 15 --weight 0.9 --batch_size 1024 --learning_rate 1e-3 --weight_decay 0 --dropout_i 0.5 --num_layers 2 --num_heads 4 --dropout_g 0.1 --max_neighbor 20 --qual_opn rotate --qual_n sum --qual_aggregate sum --epoch 300000 --eval_epoch 1000 --few_rel_aggregate mean --relation_learner gran --embedding_learner transe --shuffle_background 0 --early_stopping_patience 10 --use_neighbor 1 --use_pretrain 1 --use_in_train 1 --fine_tune 1 --few 5 --device 3 > log/wd50k_5_0.9_1024_1e-3_0.5_2_4_0.1_20_rotate_sum_sum_300000_1000_mean_gran_transe_10_1_1_1_1_1 &
```
* On F-WikiPeople
```
nohup python -u main_metarh.py --dataset F-WikiPeople --embed_dim 100 --seed 42 --embed_model HINGE --max_seq_length 17 --weight 0.9 --batch_size 1024 --learning_rate 1e-3 --weight_decay 0 --dropout_i 0.5 --num_layers 2 --num_heads 4 --dropout_g 0.1 --max_neighbor 30 --qual_opn rotate --qual_n sum --qual_aggregate sum --epoch 300000 --eval_epoch 1000 --few_rel_aggregate mean --relation_learner gran --embedding_learner transe --shuffle_background 0 --early_stopping_patience 10 --use_neighbor 1 --use_pretrain 1 --use_in_train 1 --fine_tune 1 --few 5 --device 3 > log/wikipeople_5_0.9_1024_1e-3_0.5_2_4_0.1_30_rotate_sum_sum_300000_1000_mean_gran_transe_10_1_1_1_1_1 &
```

### Datasets
* Download datasets [F-WikiPeople, F-JF17K, F-WD50K](https://drive.google.com/drive/folders/1hHmL16RvdgBHMM9VKkf7fi9AiOjDmZOp?usp=sharing)


### Pre-trained embeddings
* Download pre-trained embeddings [F-WikiPeople, F-JF17K, F-WD50K](https://drive.google.com/drive/folders/1teL-KkBpKLtRjhQZRDWw3sopHgcH4TDK?usp=sharing)
