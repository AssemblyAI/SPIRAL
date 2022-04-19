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
from tqdm.contrib.concurrent import process_map  # or thread_map
import random

import os 

import warnings
import os
import json

warnings.filterwarnings("ignore")
os.environ['TRANSFORMERS_CACHE'] = '/media/4TBNVME/cache'
os.environ['HF_DATASETS_CACHE'] = '/media/4TBNVME/cache'

import random 
import multiprocessing

parser = argparse.ArgumentParser(description='Generate training JSON from wav files')
parser.add_argument("--data_root", required=True, default=None, type=str)
parser.add_argument("--manifest_root", required=True, default=None, type=str)
parser.add_argument("--preprocess", required=False, default=False, type=bool)
parser.add_argument("--train_split", required=False, default=0.8, type=float)

args = parser.parse_args()

def __process_data(pid, data_folder, manifest_file):
    """
    Converts flac to wav and build manifests's json
    Args:
        data_folder: source with wav files
        manifest_file: where to store manifest
    Returns:
    """

    # data_folder, manifest_file = arg
    files = []
    print('reading files ', data_folder)
    for root, _, filenames in os.walk(data_folder):
        for filename in filenames:
            files.append((os.path.join(root, filename), root))
    random.shuffle(files)
    print(len(files), ' ', data_folder)
    train_entries = []
    test_entries = []
    total_files = len(files)
    count = 0
    failed_files = 0
    with tqdm(total=len(files), position=pid+1, desc=data_folder) as pbar:
        for wav_file, root in files:
            try:
                with open(wav_file, encoding="utf-8") as fin:
                    count+=1
                    # check duration
                    duration = subprocess.check_output("soxi -D {0}".format(wav_file), shell=True)

                    entry = {}
                    entry['audio_filepath'] = os.path.abspath(wav_file)
                    entry['duration'] = float(duration)
                    entry['text'] = ''
                    if count/total_files >= args.train_split:
                        train_entries.append(entry)
                    else:
                        test_entries.append(entry)
            except:
                failed_files+=1
                print('failed files :', failed_files, manifest_file)
            pbar.update(1)
    print('saving: ', manifest_file)
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
    datasets = ['dutch/downloads_ssl_nl','french/downloads_ssl_fr', 'german/downloads_ssl_de', 'italian/downloads_ssl_it','portugese/pt-br/downloads_ssl_pt_br','portugese/pt-pt/downloads_ssl_pt_pt', 'spanish/downloads_ssl_es']
    arguments = []
    for dataset in datasets:
        language = dataset.split('/')[0]
        arg = (os.path.join(data_root, dataset), os.path.join(manifest_root, language))
        arguments.append((os.path.join(data_root, dataset), os.path.join(manifest_root, language)))
        # __process_data(arg)
    pool = multiprocessing.Pool(processes=len(datasets))
    print(arguments)
    jobs = [pool.apply_async(__process_data, args=(pid, data_f, manifest_f)) for pid, (data_f,manifest_f) in enumerate(arguments)]
    pool.close()
    result_list = [job.get() for job in jobs]

    # Important to print these blanks
    print("\n" * (len(argument_list) + 1))
    # outputs = process_map(__process_data, arguments)



if __name__ == "__main__":
    main()
