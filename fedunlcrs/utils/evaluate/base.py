class BaseMetric():
    def __init__(self) -> None:
        return
    
    def reset(self) -> None:
        raise NotImplementedError
    
    def step(self) -> None:
        raise NotImplementedError
    
    def report(self) -> float:
        raise NotImplementedError