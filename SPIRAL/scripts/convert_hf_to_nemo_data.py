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

python scripts/convert_hf_to_nemo_data.py \
    output_dir=/media/AIIhot/frmccann/nemo_cv_fr \
    path=/media/AAIhot/ASR_DATA/French/french_common_voice \
    name=fr \
    split=train

python scripts/convert_hf_to_nemo_data.py \
    output_dir=/media/AIIhot/frmccann/nemo_ml_fr \
    path=/media/AAIhot/ASR_DATA/French/french_multilingual \
    name=fr \
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
from datasets import load_dataset, load_from_disk
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
import soundfile as sf
import uuid

@dataclass
class HFDatasetConvertionConfig:
    # Nemo Dataset info
    output_dir :  Optional[str] = None  # path to output directory where the manifest file will be saved

    # HF Dataset info
    path :  Optional[str] = None # HF dataset path
    name : Optional[str] = None  # name of the dataset subset
    split : Optional[str] = None  # split of the dataset subset

    # NeMo dataset conversion
    new_sr : int = 16000
    streaming : bool = False  # Whether to use Streaming dataset API. [NOT RECOMMENDED]

    # Placeholders. Generated internally.
    vocab_path : Optional[str] = None
    wav_dir : str = '/media/AAIhot/fr_cv_wavs_trial'

def prepare_output_dirs(cfg: HFDatasetConvertionConfig):
    """
    Prepare output directories and subfolders as needed.
    Also prepare the arguments of the config with these directories.
    """
    print(cfg)
    output_dir = os.path.abspath(cfg.output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)



def preprocessor(batch, wav_dir):
    audio = np.array(batch['input_values'])
    labels = batch['labels']
    
    new_audio = audio.squeeze()

    new_text = labels
    wav_filename = os.path.join(wav_dir, uuid.uuid4().hex+'.wav')
    duration = librosa.get_duration(new_audio, sr=16000)
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


    dataset = load_from_disk(cfg.path)
    dataset = dataset[cfg.split] if cfg.split else dataset
    dataset = dataset.map(functools.partial(preprocessor,wav_dir=cfg.wav_dir), num_proc=16)
    if not os.path.exists(cfg.wav_dir):
        os.makedirs(cfg.wav_dir, exist_ok=True)
    import tqdm
    import IPython
    for row in tqdm.tqdm(dataset):
        sf.write(row['audio_filepath'], np.array(row['array']), 16000)
    #     write(row['audio_filepath'], 16000, np.array(row['array']).astype(np.int16))
    dropped_columns = dataset.column_names
    dropped_columns.remove('audio_filepath')
    dropped_columns.remove('text')
    dropped_columns.remove('duration')
    dataset = dataset.remove_columns(dropped_columns)
    manifest_path = os.path.join(cfg.output_dir,'{name}.json'.format(name=cfg.name, split=cfg.split))
    dataset.to_json(manifest_path)

# Register the dataclass as a valid config
ConfigStore.instance().store(name='hfds_config', node=HFDatasetConvertionConfig)


if __name__ == '__main__':
    main()