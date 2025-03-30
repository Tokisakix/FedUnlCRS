from typing import Dict, List, Tuple

class FedUnlDataLoader:
    def __init__(self, dataset_name:str, batch_size:int, mask:Dict=None) -> None:
        raise NotImplementedError
    
    def get_edger(self) -> Tuple[Dict]:
        raise NotImplementedError

    def get_rec_data(self) -> Tuple[List, List]:
        raise NotImplementedError
    
    def get_cov_data(self) -> Tuple[List, List]:
        raise NotImplementedError