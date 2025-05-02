# TODO!
# Automatic CRS Test

MODELS = ["mlp", "hycorec", "kbrd", "bert", "gru4rec", "kgsf", "ntrd", "redial", "sasrec", "textcnn", "tgredial", "mhim", "rec"]
DATASETS = ["opendialkg", "durecdial", "hredial", "htgredial"]

REC_METRICS = ["R@10", "R@50", "M@10", "M@50", "N@10", "N@50"]
COV_METRICS = ["Dist-2", "Dist-3", "Dist-4", "BLEU-2", "BLEU-3", "BLEU-4"]
FAIRENESS_METRICS = ["APR@5", "APR@10", "APR@15", "APR@20", "LTR@5", "LTR@10", "LTR@15", "LTR@20", "Cov@5", "Cov@10", "Cov@15", "Cov@20", "Gini@5", "Gini@10", "Gini@15", "Gini@20", "KL@5", "KL@10", "KL@15", "KL@20", "Diff@5", "Diff@10", "Diff@15", "Diff@20"]

METHONS = ["random", "topk"]
ABLATION_LAYERS = ["item", "entity", "word"]
UNLEARNING_LAYERS = ["user", "conv", "item", "entity", "word", "item_hypergraph", "entity_hypergraph", "word_hypergraph"]

N_CLIENTS = [4, 8, 16, 32]
EMBEDDING_DIM = [32, 64, 128, 256]
AGGREGATE_RATE = [0.005, 0.001, 0.0005, 0.0001]