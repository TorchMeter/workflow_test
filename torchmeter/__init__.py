# Copyright (C) 2024 Ahzyuan. - All Rights Reserved
#  * You may use, distribute and modify this code under the terms of the MIT license.
#  * You should have received a copy of the MIT license with this file.
#  * If not, please visit https://rem.mit-license.org/ for more information.

"""
Torchmeter: An `all-in-one` tool for `Pytorch` model analysis, measuring:
- Params,
- FLOPs / MACs (aka. MACC or MADD), 
- Memory cost, 
- Inference time
- Throughput

Project: https://github.com/TorchMeter/torchmeter
"""

__version__ = "0.1.4-beta"

from torchmeter.core import Meter
from torchmeter.config import get_config

__all__ = ["Meter", "get_config"]