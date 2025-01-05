"""Data loader."""
import argparse
import logging
import pathlib
import pprint
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
import representation_remi
from representation_remi import Indexer
from pathlib import Path
import random
import numpy as np
import torch
import torch.utils.data
import tqdm
from utils import load_txt, save_txt
import utils
import transformers
from transformers import BloomForCausalLM
from transformers import BloomForTokenClassification
from transformers import BloomTokenizerFast
from transformers import TrainingArguments
from transformers import Trainer
from transformers import AutoTokenizer

random.seed(42)

@utils.resolve_paths
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        choices=("sod", "lmd", "lmd_full", "snd"),
        required=False,
        help="dataset key",
    )
    parser.add_argument("-n", "--names", type=pathlib.Path, help="input names")
    parser.add_argument(
        "-i", "--in_dir", type=pathlib.Path, help="input data directory"
    )
    parser.add_argument(
        "-t", "--text_dir", type=pathlib.Path, help="input data directory"
    )
    # Data
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
        default=False,
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


def pad(data, maxlen=None):
    if maxlen is None:
        max_len = max(len(x) for x in data)
    else:
        for x in data:
            assert len(x) <= max_len
    if data[0].ndim == 1:
        padded = [np.pad(x, (0, max_len - len(x))) for x in data]
    elif data[0].ndim == 2:
        padded = [np.pad(x, ((0, max_len - len(x)), (0, 0))) for x in data]
    else:
        raise ValueError("Got 3D data.")
    return np.stack(padded)


def get_mask(data):
    # padd less than max_seq_len music
    max_seq_len = max(len(sample) for sample in data)
    mask = torch.zeros((len(data), max_seq_len), dtype=torch.bool)
    for i, seq in enumerate(data):
        mask[i, : len(seq)] = 1
    return mask


class MusicDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        filename,
        data_dir,
        text_dir,
        encoding,
        max_seq_len=None,
        max_beat=None,
        use_csv=False,
        use_augmentation=False,

    ):
        super().__init__()
        self.data_dir = pathlib.Path(data_dir)
        with open(filename) as f:
            self.names = [line.strip() for line in f if line]
        self.encoding = encoding
        self.text_dir = text_dir
        self.max_seq_len = max_seq_len
        self.max_beat = max_beat
        self.use_csv = use_csv
        self.use_augmentation = use_augmentation
        self.tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")
        self.tokenizer_vocab_size = self.tokenizer.vocab_size
        self.indexer = Indexer(is_training=True)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        music_seq = self._get_music_seq(name) # input
        text_seq, attention_mask = self._get_text_seq(name) # target

        return {
            'name': name,
            'input_ids': text_seq,
            'attention_mask': attention_mask, 
            'labels': music_seq
        }

    def _get_music_seq(self, name):
        # Get the name

        # Load data
        if self.use_csv:
            notes = utils.load_csv(self.data_dir / f"{name}.csv")
        else:
            file_path = self.data_dir / name[2] / name[3]/f"{name}.npy"
            notes = np.load(file_path)

        # Check the shape of the loaded notes
        assert notes.shape[1] == 5

        # Trim sequence to max_beat
        if self.max_beat is not None:
            n_beats = notes[-1, 0] + 1
            if n_beats > self.max_beat: 
                # randomly select a starting beat
                start_beat = np.random.randint(n_beats - self.max_beat)
                end_beat = start_beat + self.max_beat
                notes = notes[(notes[:, 0] >= start_beat) & (notes[:, 0] < end_beat)]

        # get start of music and end of music within this representation
        music_seq = representation_remi.encode_notes(notes, self.encoding, self.indexer) # Encode the instruments
        # should add default dictionary size to each encoding
        # 128256
        music_seq = music_seq + self.tokenizer_vocab_size
        music_seq = torch.tensor(music_seq, dtype=torch.long) # list of tensors
        return music_seq

    def _get_text_seq(self,name):
        file_path = self.text_dir / name[2] / name[3] / f"{name}.txt"
        load_text = utils.load_txt(file_path)
        # if len(load_text) == 0:
        #     print(f"no associated description") # some of music doesn't have text descirptio, 128000: start of song
        tokenized_input = self.tokenizer(
            load_text,
            padding=False,
            truncation=True,
            max_length=36, # maximium text length, sane with caption
            return_tensors="pt"
        ) # return pytorch tensors
        input_ids = tokenized_input['input_ids'].squeeze() # squeeze, numpy
        attention_mask = tokenized_input['attention_mask'].squeeze()# will be all ones, numpy
        input_ids = input_ids.long()
        attention_mask = attention_mask.long()
        return input_ids, attention_mask

    def _get_attention_mask(self, seq):
        return np.ones_like(seq)
        
    @staticmethod
    def collate(batch):

        names = [item['name'] for item in batch]
        input_ids = [item['input_ids'] for item in batch]
        attention_masks = [item['attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]
        tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100) # ignore in loss calculation

        return {
            'name': names,
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'labels': labels
        }


def main():
    # base_model_name = "meta-llama/Meta-Llama-Guard-2-8B"
    # base_model = AutoModelForCausalLM.from_pretrained(
    #     base_model_name, 
    #     device_map="auto", 
    #     load_in_8bit=True
    # )
    # tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # # Get the embedding size from the model's configuration
    # embedding_size = base_model.config.hidden_size

    # # Get the tokenizer type
    # tokenizer_type = type(tokenizer).__name__

    # print(f"Embedding Size: {embedding_size}") # 4096
    # print(f"Tokenizer Type: {tokenizer_type}") # PreTrainedTokenizerFast

    """Main function."""
    # Parse the command-line arguments
    args = parse_args()

    # Set default arguments
    if args.dataset is not None:
        if args.names is None:
            args.names = pathlib.Path(
                f"data/{args.dataset}/processed/names.txt"
            )
        if args.in_dir is None:
            args.in_dir = pathlib.Path(f"data/{args.dataset}/processed/notes")
    if args.jobs is None:
        args.jobs = min(args.batch_size, 8)

    # Set up the logger
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(message)s",
    )

    # Log arguments
    logging.info(f"Using arguments:\n{pprint.pformat(vars(args))}")

    # Load the encoding
    encoding = representation_remi.load_encoding("/data2/weihanx/musicgpt/encoding_remi.json")

    # Create the dataset and data loader
    dataset = MusicDataset(
        args.names,
        args.in_dir,
        args.text_dir,
        encoding=encoding,
        max_seq_len=args.max_seq_len,
        max_beat=args.max_beat,
        use_csv=args.use_csv,
        use_augmentation=args.aug,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, args.batch_size, True, collate_fn=MusicDataset.collate
    )
    # Iterate through the DataLoader
    for batch in data_loader:
        print(batch)
        break

if __name__ == "__main__":
    main()
