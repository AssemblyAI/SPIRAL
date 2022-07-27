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
'''

python scripts/format_uns_data.py \
    --data_root=/media/AAIhot/yt_crawler \
    --manifest_root=/media/AAIhot/francis/manifests \


'''
import argparse
import json
import os
import subprocess

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map  # or thread_map
import random

import os 

import glob
import warnings
import os
import json

warnings.filterwarnings("ignore")
os.environ['TRANSFORMERS_CACHE'] = '/media/4TBNVME/cache'
os.environ['HF_DATASETS_CACHE'] = '/media/4TBNVME/cache'

import random 
import multiprocessing

import soundfile as sf


parser = argparse.ArgumentParser(description='Generate training JSON from wav files')
parser.add_argument("--data_root", required=True, default=None, type=str)
parser.add_argument("--manifest_root", required=True, default=None, type=str)
parser.add_argument("--train_split", required=False, default=0.999, type=float)

args = parser.parse_args()

def __process_data(pid, data_folder, manifest_file):
    """
    Converts flac to wav and build manifests's json
    Args:
        data_folder: source with wav files
        manifest_file: where to store manifest
    Returns:
    """

    files = set()
    train_entries = []
    test_entries = []

    count = len(files)
    failed_files = 0
    for filename in os.listdir(data_folder):
        wav_file = os.path.join(data_folder, filename)
        if wav_file in files or wav_file.split('.')[-1]!='wav':
            continue
        try:
            duration =sf.info(wav_file).duration
            if duration > 30:
                continue
            count+=1
            entry = {}
            entry['audio_filepath'] = os.path.abspath(wav_file)
            entry['duration'] = float(duration)
            entry['text'] = ''
            if count%1000/1000 <= args.train_split:
                train_entries.append(entry)
            else:
                test_entries.append(entry)
        except:
            failed_files+=1
            print('failed files :', failed_files, wav_file)
        if count%1000000==1:
            print('saving: ', manifest_file)
            print('count:', count)
            with open(manifest_file+'_train.json', 'w') as fout:
                for m in train_entries:
                    fout.write(json.dumps(m) + '\n')

            with open(manifest_file+'_test.json', 'w') as fout:
                for m in test_entries:
                    fout.write(json.dumps(m) + '\n')
    with open(manifest_file+'_train.json', 'w') as fout:
        for m in train_entries:
            fout.write(json.dumps(m) + '\n')

    with open(manifest_file+'_test.json', 'w') as fout:
        for m in test_entries:
            fout.write(json.dumps(m) + '\n')
    return None

def main():
    data_root = args.data_root
    manifest_root = args.manifest_root
    if not os.path.exists(os.path.dirname(manifest_root)):
        os.makedirs(os.path.dirname(manifest_root))
    datasets = ['german/downloads_ssl_de', 'italian/downloads_ssl_it','french/downloads_ssl_fr']
    arguments = []
    for dataset in datasets:
        language = dataset.split('/')[0]
        arguments.append((os.path.join(data_root, dataset), os.path.join(manifest_root, language)))
    
    pool = multiprocessing.Pool(processes=len(datasets))
    jobs = [pool.apply_async(__process_data, args=(pid, data_f, manifest_f)) for pid, (data_f,manifest_f) in enumerate(arguments)]
    pool.close()
    _ = [job.get() for job in jobs]
    print("\n" * (len(arguments) + 1))



if __name__ == "__main__":
    main()
