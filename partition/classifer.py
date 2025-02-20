import torch

class PretrainClassiferMLP(torch.nn.Module):
    def __init__(self, embedding_dim, n_item) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_item = n_item
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_dim, n_item),
            torch.nn.Softmax(dim=-1)
        )
        return
    
    def forward(self, x):
        out = self.mlp(x)
        return out

def get_classifer(classifer_model):
    classifer_model_table = {
        "mlp": PretrainClassiferMLP,
    }
    classifer = classifer_model_table[classifer_model]
    return classifer