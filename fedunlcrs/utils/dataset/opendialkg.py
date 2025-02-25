import json
from tqdm import tqdm
from typing import Dict, List

concept_net_words_list = []

def parser_dialogs(dialogs:List[Dict]):
    global concept_net_words_list
    res = []
    for dialog in dialogs:
        dialog_word = []
        for word in dialog["text"]:
            word = word.lower()
            if word in concept_net_words_list:
                dialog_word.append(word)
        dialog_word = list(set(dialog_word + dialog["word"]))
        res.append({
            "dialog_id": int(dialog["utt_id"]),
            "role": dialog["role"],
            "item": dialog["item"],
            "entity": dialog["entity"],
            "word": dialog_word,
            "text": dialog["text"],
        })
    return res

def opendialkg_dataset():
    global concept_net_words_list
    total_data = []

    with(open("data/conceptnet/en_word.txt", "r", encoding="utf-8")) as concept_net_words:
        for word in concept_net_words.readlines():
            word = word[:-1]
            concept_net_words_list.append(word)
    concept_net_words_list = set(concept_net_words_list)

    for file_name in ["train_data.json", "valid_data.json", "test_data.json"]:
        temp_data = []

        file = json.load(open(f"data/opendialkg/{file_name}", "r", encoding="utf-8"))
        for conv in file:
            temp_conv = {
                    "conv_id": int(conv["conv_id"]),
                    "user_id": conv["user_id"],
                    "dialogs": parser_dialogs(conv["dialog"]),
            }
            temp_data.append(temp_conv)

        total_data.append(temp_data)

    [train_dataset, valid_dataset, test_dataset] = total_data
    return train_dataset, valid_dataset, test_dataset