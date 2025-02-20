from .hredial import hredial_edger
from .htgredial import htgredial_edger
from .opendialkg import opendialkg_edger
from .durecdial import durecdial_edger

def get_edger(dataset):
    edger_table = {
        "hredial": hredial_edger,
        "htgredial": htgredial_edger,
        "opendialkg": opendialkg_edger,
        "durecdial": durecdial_edger,
    }
    edger = edger_table[dataset]

    return edger()