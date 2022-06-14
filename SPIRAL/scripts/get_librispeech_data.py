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
import datasets

from sox import Transformer
from tqdm import tqdm
import soundfile as sf
import IPython
parser = argparse.ArgumentParser(description='LibriSpeech Data download')
parser.add_argument("--data_root", required=False, default='/media/AAIhot/mls_spanish', type=str)
parser.add_argument("--data_sets", default="ALL", type=str)
args = parser.parse_args()

def __maybe_download_file(destination: str, source: str):
    """
    Downloads source to destination if it doesn't exist.
    If exists, skips download
    Args:
        destination: local filepath
        source: url of resource
    Returns:
    """
    source = URLS[source]
    if not os.path.exists(destination):
        logging.info("{0} does not exist. Downloading ...".format(destination))
        urllib.request.urlretrieve(source, filename=destination + '.tmp')
        os.rename(destination + '.tmp', destination)
        logging.info("Downloaded {0}.".format(destination))
    else:
        logging.info("Destination {0} exists. Skipping.".format(destination))
    return destination


def __extract_file(filepath: str, data_dir: str):
    try:
        tar = tarfile.open(filepath)
        tar.extractall(data_dir)
        tar.close()
    except Exception:
        logging.info('Not extracting. Maybe already there?')


def __process_data(data_folder: str, dst_folder: str, manifest_file: str):
    """
    Converts flac to wav and build manifests's json
    Args:
        data_folder: source with flac files
        dst_folder: where wav files will be stored
        manifest_file: where to store manifest
    Returns:
    """

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    files = []
    entries = []
    print(data_folder)
    for root, dirnames, filenames in os.walk(data_folder):
        for filename in fnmatch.filter(filenames, '*transcripts.txt'):
            files.append((os.path.join(root, filename), root))
    print(len(files))
    for transcripts_file, root in tqdm(files):
        with open(transcripts_file, encoding="utf-8") as fin:
            for line in fin:
                id, text = line[: line.index("\t")], line[line.index("\t") + 1 :]
                folders = id.split('_')
                transcript_text = text.lower().strip()

                # Convert FLAC file to WAV
                flac_file = os.path.join(root,'audio', folders[0], folders[1], id + ".flac")
                wav_file = os.path.join(dst_folder, id + ".wav")
                if not os.path.exists(wav_file):
                    Transformer().build(flac_file, wav_file)
                # check duration
                duration =sf.info(wav_file).duration
                # duration = subprocess.check_output("soxi -D {0}".format(wav_file), shell=True)

                entry = {}
                entry['audio_filepath'] = os.path.abspath(wav_file)
                entry['duration'] = float(duration)
                entry['text'] = transcript_text
                entries.append(entry)

    with open(manifest_file, 'w') as fout:
        for m in entries:
            fout.write(json.dumps(m) + '\n')


def main():
    data_root = args.data_root
    data_sets = args.data_sets

    if data_sets == "ALL":
        data_sets = "train,test,dev"
    print(data_sets.split(','))
    for data_set in data_sets.split(','):
        logging.info("Processing {0}".format(data_set))
        print(os.path.join(data_root, data_set))
        __process_data(
            os.path.join(data_root, data_set),
            os.path.join(data_root, data_set + "-processed"),
            '/home/cirrascale/frmccann/data/manifest_files' + data_set + 'es_mls' + ".json",
        )
    logging.info('Done!')


if __name__ == "__main__":
    main()
