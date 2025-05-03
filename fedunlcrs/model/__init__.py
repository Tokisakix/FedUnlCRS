import torch
from typing import List, Dict, Tuple

from .mlp import FedUnlMlp
from .hycorec import HyCoRec
from .kbrd import KBRDModel
from .kgsf import KGSFModel
from .ntrd import NTRDModel
from .redial import ReDialRecModel
from .tgredial import TGRecModel
from .rec import SASREC
from .sasrec import SASRECModel
from .gru4rec import GRU4RECModel
from .bert import BERTModel
from .textcnn import TextCNNModel
from .mhim import MHIMModel

class ABModel():
    def __init__(
            self, n_item:int, n_entity:int, n_word:int,
            model_config:Dict, device:str
        ) -> None:
        raise NotImplementedError
    
    def rec_forward(
            self, batch_data:List[Dict],
            item_edger:Dict, entity_edger:Dict, word_edger:Dict
        ) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:
        raise NotImplementedError
    
    def con_forward(
            self, batch_data:List[Dict],
            item_edger:Dict, entity_edger:Dict, word_edger:Dict
        ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.FloatTensor]:
        raise NotImplementedError