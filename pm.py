import pandas as pd
import numpy as np

from pathlib import Path

from abc import ABC, abstractmethod

## custom lib
from config import PathConfig, PMConfig

class BasePM(ABC):
    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError
    
    @property
    @abstractmethod
    def description(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def scheme(self):
        raise NotImplementedError    

    @property
    @abstractmethod
    def parameters(self):
        raise NotImplementedError
    
    


class PM(BasePM):
    name = "price_sma_momentum"
    parameters = ["sma_days"]
    load_path = PathConfig.data_path

    def __init__(self) -> None:
        super().__init__()

        