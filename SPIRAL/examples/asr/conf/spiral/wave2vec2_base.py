# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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

# from nemo.collections.asr.models.rnnt_bpe_models import RNNTBPEDecodingConfig
from spiral_nemo.collections.asr.models.configs.common_config import AudioDatasetConfig, AdamWParams, \
    CosineAnnealingParams, Conv2dBlock, Conv2dNormAct, Conv1dNormAct
from spiral_nemo.collections.asr.models.wav2vec.wav2vec_config import Wav2VecEncoderModelConfig, Wav2VecTransformerEncoderConfig, \
    Wav2VecTransformerConfig, ConvConfig, Wav2VecActivationType, QuantizerConfig, Wav2VecMaskingConfig, Wav2VecMaskType, \
    LossConfig
from spiral_nemo.collections.asr.modules.audio_preprocessing import AudioToMelSpectrogramPreprocessorConfig
from nemo.core.config import TrainerConfig
from spiral_nemo.core.config.modelPT import ModelPTConfig
from nemo.utils.exp_manager import ExpManagerConfig, CallbackParams


config_name = 'st2vec'

sample_rate = 16000
num_features = 128

max_steps=200000

model = Wav2VecEncoderModelConfig()

model.train_ds = AudioDatasetConfig(
    manifest_filepath='manifest_files/dutch_train.json,manifest_files/french_train.json,manifest_files/german_train.json,manifest_files/italian_train.json,manifest_files/spanish_train.json',
    sample_rate=sample_rate,
    batch_size=24,
    min_duration=2.0,
    crop_size=250000,
    shuffle=True,
    num_workers=12,
    pin_memory=True,
)

model.validation_ds = AudioDatasetConfig(
    manifest_filepath='manifest_files/dutch_test.json,manifest_files/french_test.json,manifest_files/german_test.json,manifest_files/italian_test.json,manifest_files/spanish_test.json',
    sample_rate=sample_rate,
    batch_size=24,
    min_duration=2.0,
    crop_size=250000,
    shuffle=False,
    num_workers=12,
)

model.test_ds = AudioDatasetConfig(
    manifest_filepath='manifest_files/dutch_test.json,manifest_files/french_test.json,manifest_files/german_test.json,manifest_files/italian_test.json,manifest_files/spanish_test.json',
    sample_rate=sample_rate,
    batch_size=24,
    min_duration=2.0,
    crop_size=250000,
    shuffle=False,
    num_workers=12,
)

model.expected_gpu_num = 8
model.optim = AdamWParams(
    lr=0.003,
    eps=1e-6,
    betas=[0.9, 0.98],
    weight_decay=0.01,
    sched=CosineAnnealingParams(
        min_lr=0.0,
        warmup_steps=32000,
        max_steps=max_steps,
    ),
)

trainer = TrainerConfig(
    gpus=1,
    max_epochs=280,
    accelerator='ddp',
    accumulate_grad_batches=1,
    checkpoint_callback=False, # Provided by exp_manager
    logger=None,  # Provided by exp_manager
    log_every_n_steps=50,
    progress_bar_refresh_rate=50,
    num_sanity_val_steps=0,
    check_val_every_n_epoch=4,
)

exp_manager = ExpManagerConfig(
    name=config_name,
    create_checkpoint_callback=True,
    checkpoint_callback_params=CallbackParams(
        save_top_k=5
    )

)

cfg = ModelPTConfig(
    name=config_name,
    model=model,
    trainer=trainer,
    exp_manager=exp_manager
)
