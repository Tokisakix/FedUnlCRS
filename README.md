# FedUnlCRS

download dataset from [link](https://drive.google.com/file/d/1VNWU6d1SRcrucAxQdDcFxRALmVQge2TJ/view?usp=sharing) and place in `data/`

```shell
data
├─conceptnet/
├─durecdial/
├─hredial/
├─htgredial/
└─opendialkg/
```

## Env Prepare

```shell
pip install -r requirements.txt
```

## Pretrain

Pretrain the embedding model which used for the dataset partition task.

```shell
python main.py --task_config config/pretrain_config.yaml --model_config config/model_config.yaml
```

## Partition

Split the dataset in kmeans algorithm.

```shell
python main.py --task_config config/partition_config.yaml --model_config config/model_config.yaml
```