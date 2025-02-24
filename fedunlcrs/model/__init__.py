from .mlp import PretrainClassiferMLP

def get_classifer(classifer_model):
    classifer_model_table = {
        "mlp": PretrainClassiferMLP,
    }
    classifer = classifer_model_table[classifer_model]
    return classifer