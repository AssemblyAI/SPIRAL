import os 

from datasets import load_dataset, load_metric, load_from_disk, concatenate_datasets
import warnings
import os
import json
import pdb
from dlutils.preprocessing.text.labels import Labels

import torchaudio
import random 
import torch
import numpy as np 


warnings.filterwarnings("ignore")

os.environ['TRANSFORMERS_CACHE'] = '/media/4TBNVME/cache'
os.environ['HF_DATASETS_CACHE'] = '/media/4TBNVME/cache'

from datasets import load_dataset, load_metric, load_from_disk, concatenate_datasets

warnings.filterwarnings("ignore")

os.environ['TRANSFORMERS_CACHE'] = '/media/4TBNVME/cache'
os.environ['HF_DATASETS_CACHE'] = '/media/4TBNVME/cache'

from datasets import load_dataset, load_metric, load_from_disk, concatenate_datasets


vocab_path = './data/es_vocab.json'

def main():
    common_voice = load_dataset("common_voice", 'es')

    common_voice_dataset = concatenate_datasets([common_voice['train'], common_voice['test'], common_voice['validation']])
    common_voice_dataset = common_voice_dataset.remove_columns(['client_id','up_votes', 'down_votes', 'age', 'gender', 'accent', 'locale', 'segment'])
    common_voice_dataset = common_voice_dataset.rename_column("sentence", "labels")
    common_voice_dataset = common_voice_dataset.rename_column("audio", "input_values")

    resample_48khz = torchaudio.transforms.Resample(48000, 16000)
    prep_labels = Labels()

    def preprocessor(batch):
        audio = torch.from_numpy(batch['input_values']['array'])
        labels = batch['labels']
        
        new_audio = resample_48khz(audio)
        new_audio = new_audio.squeeze().numpy()

        new_text = prep_labels.prep_labels(labels_path=None, labels=[labels], vocab_path=vocab_path, lang='it')[0]

        new_batch = {}

        new_batch["input_values"] = new_audio.tolist()
        new_batch["labels"] = new_text 
        new_batch['input_length'] = len(new_batch['input_values'])
        return new_batch

    common_voice = common_voice_dataset.map(preprocessor, num_proc=16)

    common_voice.save_to_disk("spanish_common_voice/")

if __name__ == "__main__":
    main()
