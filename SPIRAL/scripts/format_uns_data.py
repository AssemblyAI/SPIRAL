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
import fnmatch
import json
import logging
import os
import subprocess
import tarfile
import urllib.request

from tqdm import tqdm

parser = argparse.ArgumentParser(description='Generate training JSON from wav files')
parser.add_argument("--data_root", required=True, default=None, type=str)
parser.add_argument("--preprocess", required=False, default=False, type=bool)
args = parser.parse_args()


def __process_data(data_folder: str, manifest_file: str):
    """
    Converts flac to wav and build manifests's json
    Args:
        data_folder: source with wav files
        manifest_file: where to store manifest
    Returns:
    """

    files = []
    entries = []

    for root, dirnames, filenames in os.walk(data_folder):
        for filename in filenames:
            files.append((os.path.join(root, filename), root))

    for wav_file, root in tqdm(files):
        with open(wav_file, encoding="utf-8") as fin:
            # check duration
            duration = subprocess.check_output("soxi -D {0}".format(wav_file), shell=True)

            entry = {}
            entry['audio_filepath'] = os.path.abspath(wav_file)
            entry['duration'] = float(duration)
            entry['text'] = ''
            entries.append(entry)

    with open(manifest_file, 'w') as fout:
        for m in entries:
            fout.write(json.dumps(m) + '\n')


def main():
    data_root = args.data_root
    datasets = ['downloads_ssl_fr']
    # TODO(frmccann) add support for multiple directories 
    for dataset in datasets:
        directory = os.path.join(data_root, dataset)
        logging.info("Processing {0}".format(dataset))
        __process_data(directory,
            os.path.join(data_root, dataset + ".json"),
        )
        logging.info('Done!')


if __name__ == "__main__":
    main()
