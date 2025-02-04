# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from spiral_nemo.core.config.base_config import Config
from spiral_nemo.core.config.optimizers import (
    AdadeltaParams,
    AdagradParams,
    AdamaxParams,
    AdamParams,
    AdamWParams,
    NovogradParams,
    OptimizerParams,
    RMSpropParams,
    RpropParams,
    SGDParams,
    get_optimizer_config,
    register_optimizer_params,
)
from spiral_nemo.core.config.pytorch import DataLoaderConfig
# from spiral_nemo.core.config.pytorch_lightning import TrainerConfig, HTrainerConfig
from nemo.core.config.pytorch_lightning import TrainerConfig
from spiral_nemo.core.config.schedulers import (
    CosineAnnealingParams,
    InverseSquareRootAnnealingParams,
    NoamAnnealingParams,
    PolynomialDecayAnnealingParams,
    PolynomialHoldDecayAnnealingParams,
    SchedulerParams,
    SquareAnnealingParams,
    SquareRootAnnealingParams,
    WarmupAnnealingParams,
    WarmupHoldSchedulerParams,
    WarmupSchedulerParams,
    get_scheduler_config,
    register_scheduler_params,
)
from spiral_nemo.core.config.set_config import hydra_runner, set_config
