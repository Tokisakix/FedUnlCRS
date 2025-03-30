from typing import Dict, List, Tuple

class GraphUnlSampler:
    def __init__(self, dataset_name:str, idx_to_client:Dict) -> None:
        raise NotImplementedError
    
    def sample(self, rate:float, layer:List[str], methon:str) -> Tuple[List[int], List[int]]:
        raise NotImplementedError

class HyperGraphUnlSampler:
    def __init__(self, dataset_name:str, idx_to_client:Dict) -> None:
        raise NotImplementedError
    
    def sample(self, rate:float, layer:List[str], methon:str) -> Tuple[List[int], List[int]]:
        raise NotImplementedError