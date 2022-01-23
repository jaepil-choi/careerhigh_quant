import pandas as pd
import numpy as np

from pathlib import Path

from abc import ABC, abstractmethod

import quantstats as qs

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
    def hyperparams(self):
        raise NotImplementedError

class DefaultPM(BasePM):
    def __init__(self) -> None:
        super().__init__()


    def plot_result(self, returns):
        qs.reports.full(returns, benchmark="SPY")


class PM(DefaultPM):
    name = "price_sma_momentum"
    description = "Simple price / SMA crossover momentum for a single asset"
    hyperparams = ["sma_days"]
    load_path = PathConfig.data_path
    scheme = "D-0"

    def __init__(
        self, 
        pm_config,
        ) -> None:

        super().__init__()
        for k, v in pm_config.items():
            # if k in classvars
            setattr(PM, k, v)

            # if k in instancevars
            setattr(self, k, v)
            
        
        
        