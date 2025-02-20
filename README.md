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

## Partition

```shell
python main.py --task partition
```