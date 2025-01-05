import torch
import subprocess
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig, AutoConfig
from accelerate import PartialState
import wandb
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from peft import PeftConfig
import json
import copy 
import huggingface_hub
import safetensors
from collections import defaultdict
from datasets import load_dataset
import pathlib
from torch.utils.data.dataloader import default_collate
import tqdm
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import representation_remi
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW, get_scheduler, Trainer, TrainingArguments
from dataset_textmusic_bloom import MusicDataset
from utils import load_txt
import argparse
import logging
import sys
import transformers
import linear_attention_transformer
import matplotlib.pyplot as plt
import numpy as np
import performer_pytorch
import torch
import torch.utils.data
import tqdm
import x_transformers
from peft import PeftModel
import music_x_transformers
import utils
import muspy 
import numpy as np
import torch
import torch.utils.data
import tqdm
import os
from peft import AutoPeftModelForCausalLM
from transformers import BloomForCausalLM
from transformers import BloomForTokenClassification
from transformers import BloomTokenizerFast
from transformers import TrainingArguments
from transformers import Trainer
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import torch
from datasets import load_dataset
import random
sys.path.append("/data2/weihanx/musicgpt")
torch.cuda.empty_cache()

def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    # removed negative pitch file
    parser.add_argument("-nt", "--test_names", type=str, help="test names", default=Path("/data2/weihanx/musicgpt/mm_ft_file/test-names.txt")) 
    parser.add_argument(
        "-i", "--in_dir", type=pathlib.Path, help="note data directory", default=Path("/data2/weihanx/musicgpt/combined_dataset/musescore_notes")
    )
    parser.add_argument(
        "-t", "--text_dir", type=pathlib.Path, help="text data directory", default=Path("/data2/weihanx/musicgpt/metascore_genre/metascore_genre_tag_cap")
    )
    parser.add_argument(
        "-o", "--out_dir", type=pathlib.Path, help="output directory", default=Path("/data2/weihanx/musicgpt/bloom_generated")
    )

    parser.add_argument(
        "-rep", 
        "--representation", 
        choices=("remi", "tag"),
        default="remi",
        required=False,
    )
    # Data
    parser.add_argument(
        "--steps",
        default=500000,
        type=int,
        help="number of steps",
    )
    parser.add_argument(
        "--valid_steps",
        default=10000,
        type=int,
        help="validation frequency",
    )
    parser.add_argument(
        "--early_stopping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to use early stopping",
    )
    parser.add_argument(
        "-e",
        "--early_stopping_tolerance",
        default=10,
        type=int,
        help="number of extra validation rounds before early stopping",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        default=0.0005,
        type=float,
        help="learning rate",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        default=5000,
        type=int,
        help="learning rate warmup steps",
    )
    parser.add_argument(
        "--lr_decay_steps",
        default=100000,
        type=int,
        help="learning rate decay end steps",
    )
    parser.add_argument(
        "--lr_decay_multiplier",
        default=0.1,
        type=float,
        help="learning rate multiplier at the end",
    )
    parser.add_argument(
        "--grad_norm_clip",
        default=1.0,
        type=float,
        help="gradient norm clipping",
    )
    parser.add_argument(
        "--weight_decay",
        default=0.01,
        type=float,
        help="weight decay",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        default=8,
        type=int,
        help="batch size",
    )
    parser.add_argument(
        "--use_csv",
        action="store_true",
        help="whether to save outputs in CSV format (default to NPY format)",
    )
    parser.add_argument(
        "--aug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to use data augmentation",
    )
    parser.add_argument(
        "--max_seq_len",
        default=1024,
        type=int,
        help="maximum sequence length",
    )
    parser.add_argument(
        "--max_beat",
        default=256,
        type=int,
        help="maximum number of beats",
    )
    parser.add_argument(
    "--grad_acc",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="whether to accumulate gradients to increase the batch size",
)
    # Others
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        help="number of jobs (deafult to `min(batch_size, 8)`)",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="show warnings only"
    )
    return parser.parse_args(args=args, namespace=namespace)

def get_lr_multiplier(
    step, warmup_steps, decay_end_steps, decay_end_multiplier
):
    """Return the learning rate multiplier with a warmup and decay schedule.

    The learning rate multiplier starts from 0 and linearly increases to 1
    after `warmup_steps`. After that, it linearly decreases to
    `decay_end_multiplier` until `decay_end_steps` is reached.

    """
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if step > decay_end_steps:
        return decay_end_multiplier
    position = (step - warmup_steps) / (decay_end_steps - warmup_steps)
    return 1 - (1 - decay_end_multiplier) * position


# test_dataset = MusicDataset(
#     args.test_names,
#     args.in_dir,
#     args.text_dir,
#     encoding=encoding,
#     max_seq_len=args.max_seq_len,
#     max_beat=args.max_beat,
#     use_csv=args.use_csv,
#     use_augmentation=False,
# )

# # print(f"test dataset = {test_dataset[0]}")
# device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# sample_dir = args.out_dir
# (sample_dir / "npy").parent.mkdir(exist_ok=True)
# (sample_dir / "csv").parent.mkdir(exist_ok=True)
# (sample_dir / "txt").parent.mkdir(exist_ok=True)
# (sample_dir / "json").parent.mkdir(exist_ok=True)
# (sample_dir / "png").parent.mkdir(exist_ok=True)
# (sample_dir / "mid").mkdir(exist_ok=True)
# (sample_dir / "wav").mkdir(exist_ok=True)
# (sample_dir / "mp3").mkdir(exist_ok=True)
# (sample_dir / "png-trimmed").mkdir(exist_ok=True)
# (sample_dir / "wav-trimmed").mkdir(exist_ok=True)
# (sample_dir / "mp3-trimmed").mkdir(exist_ok=True)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")

representation = representation_remi
old_num_vocab = 250880
encoding = representation.load_encoding("/data2/weihanx/musicgpt/encoding_remi.json")
event_code_map = encoding["event_code_map"]
start_of_song = event_code_map["start-of-song"] + old_num_vocab
end_of_song = event_code_map["end-of-song"] + old_num_vocab
print(f"start of song = {start_of_song}, end of song = {end_of_song}")
results = defaultdict(list)
# test_name = utils.load_txt(args.test_names)
test_name = [
    "QmXsYQo5xjms7HFc7Ewdzwqv2y88pqv9zHYy717KpSg6s9",
    "QmVCrPS3p6BqYjofzjdBgqVBZWK5T7kMG4rgGQMa2y5Nvt",
    "QmaBZHmLcDqzv7zDzFJAb8ZCgdF6uM5gKPqBQXv6HbjBUD",
    "QmS2Brqe1LW1EwaEFdRm2weM3YUJfiP75dw3V8Mjo7jr6F",
    "QmRMUirzkktd5meWhXCm44X4eNcNSY67DXWiqu2eukAmg9",
    "QmXsYQo5xjms7HFc7Ewdzwqv2y88pqv9zHYy717KpSg6s9",
    "QmVCrPS3p6BqYjofzjdBgqVBZWK5T7kMG4rgGQMa2y5Nvt",
    "QmaBZHmLcDqzv7zDzFJAb8ZCgdF6uM5gKPqBQXv6HbjBUD",
    "QmS2Brqe1LW1EwaEFdRm2weM3YUJfiP75dw3V8Mjo7jr6F",
    "QmRMUirzkktd5meWhXCm44X4eNcNSY67DXWiqu2eukAmg9",
]

# peft_model_id ='/data2/weihanx/musicgpt/mm_ft_0830/checkpoint-230000'

# reference:https://huggingface.co/docs/transformers/peft
model_id = "bigscience/bloom-560m"
adapter_model_id = "/data2/weihanx/musicgpt/bloom_ft_1220_8"

model = AutoModelForCausalLM.from_pretrained(model_id)
model.resize_token_embeddings(252416) # add new tokens in the end
model_without_adapter = copy.deepcopy(model)
# model.load_adapter(adapter_model_id)
# Iterate over parameters and compare
# Iterate over parameters and compare, excluding the embedding layer if necessary
print("Comparing model parameters")
print(f"model = {model}")

# print("LoRA Embedding A Parameters:", model.transformer.word_embeddings.lora_embedding_A) # alreay check, not empty
# print("LoRA Embedding B Parameters:", model.transformer.word_embeddings.lora_embedding_B)
# print(f"model is loaded = {model}") # word embeding: 252416, 1024
# # If you have additional states like the optimizer or scheduler to load:
# optimizer_state_dict = torch.load(f"{checkpoint_dir}/optimizer.pt")
# scheduler_state_dict = torch.load(f"{checkpoint_dir}/scheduler.pt")
# print(f"model is loaded = {model}") # word embeding: 252416, 1024
## for generation

model_without_adapter.eval()
model_without_adapter.to(device)
count = 0
with torch.no_grad():
    for music in test_name:
        count += 1
        pad_token_id = tokenizer.pad_token_id  # Padding token ID
        sos_token_id = tokenizer.bos_token_id  # Start-of-sequence token ID
        eos_token_id = tokenizer.eos_token_id  # End-of-sequence token ID

        # TODO: MIGHT NEED UGT
        # # print(f"music = {music}")
        # text_description = args.text_dir  / music[2] / music[3] / f"{music}.txt"
        # text_description = utils.load_txt(text_description)[0]
        # print(f"Text:{text_description}")
        text_description = "A piece of music."
        # if not text_description:  # This checks if text_description is empty or None
        #     continue
        # add start of song?
        codes = [sos_token_id]

        input_ids = tokenizer.encode(text_description, return_tensors="pt").to(device)

        codes.extend(input_ids[0].cpu().numpy().tolist())

        # # print(f"codes = {codes}")
        codes.extend([start_of_song])
        input_ids = torch.tensor(codes, device=device).unsqueeze(0)
        # print(f"input_ids = {input_ids}")
        # start_song = torch.tensor([[start_of_song]], device=device)  # Must match the shape or be broadcastable
        # input_ids = torch.cat([input_ids, start_song], dim=1)
        outputs = model.generate(
            input_ids,
            max_length=32,
            output_scores=True,
            return_dict_in_generate=True,
            temperature=1.2,  # Increase randomness
            do_sample=True,
            repetition_penalty=1.2,  # Penalize repetitions
            top_k=50,  # Consider top 50 tokens only
            top_p=0.95  # Consider tokens with cumulative probability > 0.95
        )
        generated_ids = outputs.sequences
        generated_ids  = generated_ids[0].cpu().numpy() - 250880 # minus start of song
        print(f"generated_ids = {generated_ids[:32]}")
        # if count > 10:
        #     break
        # try:
        #     # start_index = (output_ids[0] == start_of_song).nonzero(as_tuple=True)
        #     # print(f"start index = {start_index}")
        #     extracted_ids = output_ids[0].cpu().numpy() 
        #     # use music decoding
        #     extracted_ids = extracted_ids - old_num_vocab
        #     save_result(
        #         music,
        #         extracted_ids,
        #         sample_dir,
        #         encoding,
        #         vocabulary,
        #         representation,
        #     )
        # except Exception as e:
        #     print(f"Error: {e}")
        