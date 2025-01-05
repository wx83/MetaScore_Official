"""Data loader."""
import argparse
import logging
import pathlib
import pprint
import sys
import json

import torch
import pickle
import torch.utils.data
import numpy as np
import representation_mmm
import representation_remi
import utils
import pandas as pd
import os
from torch.utils.data import ConcatDataset

# dataset_flat call representation_remi.py
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
    parser.add_argument(
        "-r",
        "--representation",
        choices=("mmm", "remi"),
        required=True,
        help="representation key",
    )
    parser.add_argument("-n", "--names", type=pathlib.Path, help="input names")
    parser.add_argument(
        "-i", "--in_dir", type=pathlib.Path, help="input data directory 1"
    )
    parser.add_argument(
        "-it", "--in_dir_tag", type=pathlib.Path, help="input tag directory 1"
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
        default=64,
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

# For Generation
# def pad(data, maxlen=None):
#     # print(f"length of input = {len(data)}")
#     if maxlen is None:
#         max_len = max(len(x) for x in data)
#     else:
#         for x in data:
#             assert len(x) <= max_len
#     if data[0].ndim == 1:
#         padded = [np.pad(x, (0, max_len - len(x))) for x in data]
#     elif data[0].ndim == 2:
#         padded = [np.pad(x, ((0, max_len - len(x)), (0, 0))) for x in data]
#     else:
#         raise ValueError("Got 3D data.")
#     # print(f"length of output = {len(padded)}")
#     return np.stack(padded)
# For classication: Need to be divisable by window size
def pad(data, maxlen=None):
    # print(f"length of input = {len(data.shape)}")
    window_size = 128  # Set this to your window size

    if maxlen is None:
        max_len = max(len(x) for x in data)
        # Adjust max_len to be divisible by the window size
        if max_len % window_size != 0:
            max_len = ((max_len // window_size) + 1) * window_size
    else:
        max_len = maxlen
        assert max_len % window_size == 0, "Max length must be divisible by window size"

    if data[0].ndim == 1:
        padded = [np.pad(x, (0, max_len - len(x))) for x in data]
    elif data[0].ndim == 2:
        padded = [np.pad(x, ((0, max_len - len(x)), (0, 0))) for x in data]
    else:
        raise ValueError("Got 3D data.")
    # print(f"length of output = {len(padded.shape)}")
    return np.stack(padded)


def get_mask(data):
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
        tag_dir,
        encoding,
        indexer,
        encode_fn,
        max_seq_len=None,
        max_beat=None,
        use_csv=False,
        use_pitch_augmentation=False,
        use_beat_augmentation=False,
        tags = None # this version don't have program prefix

    ):
    # the filename should have no path at all
        # print(f"input file name = {filename}") # /data2/weihanx/musicgpt/data/genre_all/train-names.txt
        super().__init__()
        self.data_dir = pathlib.Path(data_dir)
        self.tag_dir = pathlib.Path(tag_dir)
        with open(filename) as f:
            self.names = [line.strip() for line in f if line]
        self.encoding = encoding
        self.indexer = indexer
        self.encode_fn = encode_fn
        self.max_seq_len = max_seq_len
        self.max_beat = max_beat
        self.use_csv = use_csv
        self.use_pitch_augmentation = use_pitch_augmentation
        self.use_beat_augmentation = use_beat_augmentation
        self.tags = tags
        self.filename = filename
        self._i = -1
        with open('saved_musescore_map.pkl', 'rb') as f:
            self.mapping = pickle.load(f) # the dictionary order matches

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        # Get the name
        name = self.names[idx]
        # print(f"name = {name} idx = {idx} self file name = {self.filename}")
        name = name.removesuffix('.csv')
        # Load data
        # if name start with Qm should be in musescore


        if self.use_csv:
            notes = utils.load_csv(self.data_dir / f"{name}.csv")
        else:
            if name.startswith("Qm"):
                directory = "musescore_notes"
                file_path = self.data_dir /directory/ name[2] / name[3]/f"{name}.npy"
                notes = np.load(file_path)
            else:
                directory = "topmag_note"
                notes = np.load(self.data_dir / directory / name[0]/f"{name}.npy") # if in another dataset

        # Check the shape of the loaded notes
        assert notes.shape[1] == 5

        # Shift all the pitches for k semitones (k~Uniform(-5, 6))
        if self.use_pitch_augmentation:
            pitch_shift = np.random.randint(-5, 7)
            notes[:, 2] = np.clip(notes[:, 2] + pitch_shift, 0, 127)

        # Randomly select a starting beat: first version set to false
        if self.use_beat_augmentation:
            n_beats = notes[-1, 0] + 1
            if n_beats > self.max_beat:
                trial = 0
                # Avoid section with too few notes
                while trial < 10:
                    start_beat = np.random.randint(n_beats - self.max_beat)
                    end_beat = start_beat + self.max_beat
                    sliced_notes = notes[
                        (notes[:, 0] >= start_beat) & (notes[:, 0] < end_beat)
                    ]
                    if len(sliced_notes) > 10:
                        break
                    trial += 1
                sliced_notes[:, 0] = sliced_notes[:, 0] - start_beat
                notes = sliced_notes

        # Trim sequence to max_beat
        elif self.max_beat is not None:
            n_beats = notes[-1, 0] + 1
            if n_beats > self.max_beat:
                notes = notes[notes[:, 0] < self.max_beat]

        if self.tags!=None:
            if name.startswith('Qm'):
                directory = "musescore_tag"
                file_path = self.tag_dir / directory / name[2] / name[3]/f"{name}.txt"
                tag_path = file_path
            else:
                directory = "topmag_tag_new"
                tag_path = self.tag_dir / directory / name[0]/f"{name}.txt" # if in another dataset


            # print()
            # tag = np.load(self.tag_dir / name[2] / name[3]/f"{name}.txt", allow_pickle=True)
            with open(tag_path, "r") as file:
                tag = file.readlines()
            tag_list = [i.strip() for i in tag]
            # print(f"tag list = {tag_list}")
            # need to switch back to tag label: 
            # load the label index: label index 
            # tag_map = {'r&b, funk & soul': 0, 'rock': 1, 
            # 'folk': 2, 'metal': 3, 'new age': 4, 'world music': 5, 
            # 'classical': 6, 'religious music': 7, 'blues': 8, 'pop': 9, 
            # 'jazz': 10, 'darkwave': 11, 'comedy': 12, 'soundtrack': 13, 
            # 'electronic': 14, 'hip hop': 15, 'reggae & ska': 16, 'disco': 17, 
            # 'experimental': 18, 'country': 19}
            # never need the tags
            tag_map = {'classical': 0, 'soundtrack': 1, 'religious music': 2, 'folk': 3, 'pop_rock': 4, 
            'hip hop': 5, 'metal': 6, 'world music': 7, 'r&b, funk & soul': 8, 
            'electronic': 9, 'reggae & ska': 10, 'country': 11, 'jazz': 12, 'comedy': 13, 
            'blues': 14, 'new age': 15, 'disco': 16, 'darkwave': 17, 'experimental': 18}
            seq = self.encode_fn(notes, self.encoding, self.indexer,self.tags,None)
            if self.max_seq_len is not None and len(seq) > self.max_seq_len:
                seq = np.concatenate((seq[: self.max_seq_len - 2], seq[-2:]))
            # probably multitag, should be a numpy array
            sample_labels = [tag_map[i] for i in tag_list]
            # set the numercial tag to be class labels:
            one_hot_encoded = np.zeros(19, dtype=int)
            one_hot_encoded[sample_labels] = 1
            # print(f"label = {sample_labels}, one_hot_encode = {one_hot_encoded}")

        if self.tags == None:
            ## No tag, tag=None
            seq = self.encode_fn(notes, self.encoding, self.indexer,self.tags, None) # use remi, tag is not used
            if self.max_seq_len is not None and len(seq) > self.max_seq_len:
                seq = np.concatenate((seq[: self.max_seq_len - 2], seq[-2:]))
        # print(f"name = {name}")
        return {"name": name, "seq": seq, "label": one_hot_encoded} # in numpy 

    @classmethod
    def collate(cls, data):

        seq = [sample["seq"] for sample in data]
        return {
            "name": [sample["name"] for sample in data],
            "seq": torch.tensor(pad(seq), dtype=torch.long),
            "seq_len": torch.tensor([len(s) for s in seq], dtype=torch.long),
            "mask": get_mask(seq),
            "label": [sample["label"] for sample in data]
        }


def main():
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
    if args.representation == "mmm":
        representation = representation_mmm
    elif args.representation == "remi":
        representation = representation_remi

    # Set up the logger
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(message)s",
    )

    # Log arguments
    logging.info(f"Using arguments:\n{pprint.pformat(vars(args))}")

    # Load the encoding
    encoding = representation.get_encoding()

    # Get the indexer
    indexer = representation.Indexer(encoding["event_code_map"])
    # Create the dataset and data loader
    # args.names need to be sampled, pass name of the samples into dataset
    dataset= MusicDataset(
        args.names,
        args.in_dir,
        args.in_dir_tag,
        encoding=encoding,
        indexer=indexer,
        encode_fn=representation.encode_notes,
        max_seq_len=args.max_seq_len,
        max_beat=args.max_beat,
        use_csv=args.use_csv,
        tags = "genres"
        # use_augmentation=args.aug, # not use afterwards, comment out
    )
    # dataset_2 = MusicDataset(
    #     args.names_2,
    #     args.in_dir_2,
    #     args.in_dir_tag_2,
    #     encoding=encoding,
    #     indexer=indexer,
    #     encode_fn=representation.encode_notes,
    #     max_seq_len=args.max_seq_len,
    #     max_beat=args.max_beat,
    #     use_csv=args.use_csv,
    #     tags = "genres"
    #     # use_augmentation=args.aug, # not use afterwards, comment out
    # )

    # # combind two datasets
    # dataset = ConcatDataset([dataset_1, dataset_2])

    data_loader = torch.utils.data.DataLoader(
        dataset, args.batch_size, True, collate_fn=MusicDataset.collate
    )

    # Iterate over the loader
    n_batches = 0
    n_samples = 0
    seq_lens = []
    # first element in a dataloader
    # print(f"first sequence in a dataloader{next(iter(data_loader))['seq']}")
    for batch in data_loader:
        print(f"name = {batch['name']}")
        print(f"seq = {batch['seq']}")
        print(f"label = {batch['label']}")
        break
    # for i, batch in enumerate(data_loader):
    #     # print(f"batch = {batch}")
    #     n_batches += 1
    #     # if batch == None:
    #     #     continue
    #     n_samples += len(batch["name"])

    #     seq_lens.extend(int(l) for l in batch["seq_len"])
    #     if i == 0:
    #         logging.info("Example:")
    #         for key, value in batch.items():
    #             if key == "name":
    #                 continue
    #             logging.info(f"Shape of {key}: {value.shape}")
    #         logging.info(f"Name: {batch['name'][0]}")
        
    # logging.info(
    #     f"Successfully loaded {n_batches} batches ({n_samples} samples)."
    # )

    # # Print sequence length statistics
    # logging.info(f"Avg sequence length: {np.mean(seq_lens):2f}")
    # logging.info(f"Min sequence length: {min(seq_lens)}")
    # logging.info(f"Max sequence length: {max(seq_lens)}")


if __name__ == "__main__":

    #         if keyword not in keywordList:
    #             keywordList.append(keyword)
    # pprint(keywordList)
    main()
