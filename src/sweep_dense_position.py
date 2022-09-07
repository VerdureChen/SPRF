import itertools
import json
import math
import os
import shlex
import sys
from pathlib import Path
from typing import Literal, Optional, Union

from jsonargparse import CLI
from ray import air, tune

sys.path.append(str(Path(__file__).parents[1]))

from src.Dense_Select_PRF.__main__ import

assert Path(".git").exists()
os.environ["PL_DISABLE_FORK"] = "1"