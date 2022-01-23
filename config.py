from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

from pathlib import Path
import sys

BASE_PATH = Path(__file__).resolve()

@dataclass
class PathConfig:
    data_path: Union[str, Path] = field(
        default=BASE_PATH / "mentor_materials" / "data",
    )

@dataclass
class PMConfig:
    