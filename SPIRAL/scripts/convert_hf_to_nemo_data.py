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

"""
Python wrapper over HuggingFace Datasets to create preprocessed NeMo ASR Datasets.

List of HuggingFace datasets : https://huggingface.co/datasets
(Please filter by task: automatic-speech-recognition)

# Setup
After installation of huggingface datasets (pip install datasets), some datasets might require authentication
- for example Mozilla Common Voice. You should go to the above link, register as a user and generate an API key.

## Authenticated Setup Steps

Website steps:
- Visit https://huggingface.co/settings/profile
- Visit "Access Tokens" on list of items.
- Create new token - provide a name for the token and "read" access is sufficient.
  - PRESERVE THAT TOKEN API KEY. You can copy that key for next step.
- Visit the HuggingFace Dataset page for Mozilla Common Voice
  - There should be a section that asks you for your approval.
  - Make sure you are logged in and then read that agreement.
  - If and only if you agree to the text, then accept the terms.

Code steps:
- Now on your machine, run `huggingface-cli login`
- Paste your preserved HF TOKEN API KEY (from above).

Now you should be logged in. When running the script, dont forget to set `use_auth_token=True` !

# Usage
The script supports two modes, but the offline mode is the preferred mechanism. The drawback of the offline mode
is that it requires 3 copies of the dataset to exist simultanously -

1) The .arrow files for HF cache
2) The extracted dataset in HF cache
3) The preprocessed audio files preserved in the output_dir provided in the script.

Due to this, make sure your HDD is large enough to store the processed dataset !

## Usage - Offline Mode

python convert_hf_dataset_to_nemo.py \
    output_dir=frmccann/data/nemo_cv_es \
    path=.cache/huggingface/datasets/common_voice/es/6.1.0/d3d5467c15716a2699f2ea3710fdc8bed7c20ae8ed66c248185735a0695dcc3b \
    use_auth_token=False \
    name=es \
    split=train

This will create an output directory of multiple sub-folders containing the preprocessed .wav files,
along with a nemo compatible JSON manifest file.

NOTE:
    The JSON manifest itself is not preprocessed ! You should perform text normalization, and cleanup
    inconsistent text by using NeMo Text Normalization tool and Speech Data Explorer toolkit !

## Usage - Streaming Mode

NOTE:
    This mode is not well supported. It trades of speed for storage by only having one copy of the dataset in
    output_dir, however the speed of processing is around 10x slower than offline mode. Some datasets (such as MCV)
    fail to run entirely.

    DO NOT USE if you have sufficient disk space.

python convert_hf_dataset_to_nemo.py \
    ... all the arguments from above \
    streaming=True

"""

import os
from dataclasses import dataclass, is_dataclass
from typing import Optional

import hydra
import librosa
import tqdm
from datasets import load_dataset
import torchaudio
from omegaconf import OmegaConf
import functools
import os
from dlutils.preprocessing.text.labels import Labels
import torch
import numpy as np 
from hydra.core.config_store import ConfigStore
import IPython
from scipy.io.wavfile import write
import uuid

@dataclass
class HFDatasetConvertionConfig:
    # Nemo Dataset info
    output_dir: str  # path to output directory where the manifest file will be saved

    # HF Dataset info
    path: str  # HF dataset path
    name: Optional[str] = None  # name of the dataset subset
    split: Optional[str] = None  # split of the dataset subset

    # NeMo dataset conversion
    old_sr: int = 48000
    new_sr: int = 16000
    streaming: bool = False  # Whether to use Streaming dataset API. [NOT RECOMMENDED]

    # Placeholders. Generated internally.
    vocab_path: Optional[str] = None
    wav_dir: str = '/home/cirrascale/frmccann/data/es_cv_wavs'

def prepare_output_dirs(cfg: HFDatasetConvertionConfig):
    """
    Prepare output directories and subfolders as needed.
    Also prepare the arguments of the config with these directories.
    """
    output_dir = os.path.abspath(cfg.output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)



def preprocessor(batch, resampler, label_prepper, vocab_path, language, wav_dir):
    audio = torch.from_numpy(batch['audio']['array'])
    labels = batch['sentence']
    
    new_audio = resampler(audio)
    new_audio = new_audio.squeeze().numpy()

    new_text = label_prepper.prep_labels(labels_path=None, labels=[labels], vocab_path=vocab_path, lang=language)[0]
    wav_filename = os.path.join(wav_dir, uuid.uuid4().hex+'.wav')
    relative_wav_filename = os.path.join('frmccann/data/es_cv_wavs/', uuid.uuid4().hex+'.wav')
    duration = librosa.get_duration(new_audio, sr=resampler.new_freq)
    new_batch = {}
    new_batch["audio_filepath"] = wav_filename
    new_batch["text"] = new_text 
    new_batch["duration"] = duration
    new_batch["array"] = new_audio
    return new_batch


@hydra.main(config_name='hfds_config', config_path=None)
def main(cfg: HFDatasetConvertionConfig):
    print('start')
    print(cfg.output_dir)
    os.environ['HF_DATASETS_CACHE'] = '/media/4TBNVME/cache'

    # Convert dataclass to omegaconf
    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    # Prepare output subdirs
    prepare_output_dirs(cfg)


    dataset = load_dataset(
        path=cfg.path,
        name=cfg.name,
        split=cfg.split,
        cache_dir=None,
        streaming=cfg.streaming,
    )
    # dataset = dataset.select(range(0,80))
    resample_48khz = torchaudio.transforms.Resample(cfg.old_sr, cfg.new_sr)
    prep_labels = Labels()
    dataset = dataset.map(functools.partial(preprocessor,resampler=resample_48khz, label_prepper=prep_labels, vocab_path=cfg.vocab_path, language=cfg.name, wav_dir=cfg.wav_dir), num_proc=16)
    if not os.path.exists(cfg.wav_dir):
        os.makedirs(cfg.wav_dir, exist_ok=True)
    for row in dataset:
        write(row['audio_filepath'], cfg.new_sr, np.array(row['array']).astype(np.int16))
    dropped_columns = dataset.column_names
    dropped_columns.remove('audio_filepath')
    dropped_columns.remove('text')
    dropped_columns.remove('duration')
    dataset = dataset.remove_columns(dropped_columns)
    manifest_path = os.path.join(cfg.output_dir,'cv_{name}_{split}.json'.format(name=cfg.name, split=cfg.split))
    dataset.to_json(manifest_path)

# Register the dataclass as a valid config
ConfigStore.instance().store(name='hfds_config', node=HFDatasetConvertionConfig)


if __name__ == '__main__':
    main()