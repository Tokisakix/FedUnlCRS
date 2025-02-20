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