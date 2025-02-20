from tqdm import tqdm

from .hredial import hredial_dataset
from .htgredial import htgredial_dataset
from .opendialkg import opendialkg_dataset
from .durecdial import durecdial_dataset

def get_dataset(dataset):
    dataset_table = {
        "hredial": hredial_dataset,
        "htgredial": htgredial_dataset,
        "opendialkg": opendialkg_dataset,
        "durecdial": durecdial_dataset,
    }
    dataset = dataset_table[dataset]
    return dataset()

def get_dataloader(train_dataset, item2idx, entity2idx, word2idx):
    tot_data = []

    for conv in tqdm(train_dataset):
        dialog_item = []
        dialog_entity = []
        dialog_word = []
        for dialog in conv["dialogs"]:
            if dialog["role"] == "Recommender":
                for target_item in dialog["item"]:
                    if target_item not in item2idx:
                        continue
                    item_list = []
                    for item in set(dialog_item):
                        if item in item2idx:
                            item_list.append(item2idx[item])
                    entity_list = []
                    for entity in set(dialog_entity):
                        if entity in entity2idx:
                            entity_list.append(entity2idx[entity])
                    word_list = []
                    for word in set(dialog_word):
                        if word in word2idx:
                            word_list.append(word2idx[word])
                    tot_data.append({
                        "item": item_list,
                        "entity": entity_list,
                        "word": word_list,
                        "label": [item2idx[target_item]],
                    })
            dialog_item += dialog["item"]
            dialog_entity += dialog["entity"]
            dialog_word += dialog["word"]

    return tot_data