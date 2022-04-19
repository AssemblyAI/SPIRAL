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
#
# USAGE: python get_librispeech_data.py --data_root=<where to put data>
#        --data_set=<datasets_to_download>
# where <datasets_to_download> can be: dev_clean, dev_other, test_clean,
# test_other, train_clean_100, train_clean_360, train_other_500 or ALL
# You can also put more than one data_set comma-separated:
# --data_set=dev_clean,train_clean_100
import argparse
import json
import os
import subprocess

from tqdm import tqdm

import os 

from datasets import load_from_disk
import warnings
import os
import json
import pdb
import IPython

warnings.filterwarnings("ignore")
os.environ['TRANSFORMERS_CACHE'] = '/media/4TBNVME/cache'
os.environ['HF_DATASETS_CACHE'] = '/media/4TBNVME/cache'

sampling_rate = 16000
parser = argparse.ArgumentParser(description='Generate training JSON from wav files')
parser.add_argument("--data_root", required=False, default='/media/NFS-mid/ASR_DATA/Spanish/spanish_common_voice/', type=str)
parser.add_argument("--manifest_path", required=False, default='./frmccann/data/manifest_files/spanish_cv_', type=str)
parser.add_argument("--train_split", required=False, default=0.8, type=float)
args = parser.parse_args()

def __process_data(dataset, manifest_file: str):
    """
    Converts flac to wav and build manifests's json
    Args:
        data_folder: source with wav files
        manifest_file: where to store manifest
    Returns:
    """
    total_files = len(dataset)
    count = 0
    train_entries = []
    test_entries = []
    for row in tqdm(dataset):
        count+=1
        IPython.embed()
        wav_file = row['path']
        with open(wav_file, encoding="utf-8") as fin:
            count+=1
            # check duration
            entry = {}
            entry['wav'] = row['input_values']
            entry['duration'] = row['input_length']/sampling_rate
            entry['text'] = row['labels']
            if count/total_files >= args.train_split:
                train_entries.append(entry)
            else:
                test_entries.append(entry)

    with open(manifest_file+'train.json', 'w') as fout:
        for m in train_entries:
            fout.write(json.dumps(m) + '\n')

    with open(manifest_file+'test.json', 'w') as fout:
        for m in test_entries:
            fout.write(json.dumps(m) + '\n')


def main():
    data_root = args.data_root
    dataset = load_from_disk(data_root)

    __process_data(dataset, args.manifest_path)


if __name__ == "__main__":
    main()
