"""Data loader."""
import argparse
import logging
import pathlib
import pprint
import sys
import json
import numpy as np
import torch
import pickle
import torch.utils.data

# import representation_mmm
import representation_remi
import utils
import pandas as pd
import os
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
        "-i", "--in_dir", type=pathlib.Path, help="input data directory"
    )
    parser.add_argument(
        "-it", "--in_dir_tag", type=pathlib.Path, help="input tag directory"
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
        tags = None

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

        if self.use_csv:
            notes = utils.load_csv(self.data_dir / f"{name}.csv")
        else:
            notes = np.load(self.data_dir / name[2] / name[3]/f"{name}.npy")

        # Check the shape of the loaded notes
        assert notes.shape[1] == 5

        # Shift all the pitches for k semitones (k~Uniform(-5, 6))
        if self.use_pitch_augmentation:
            pitch_shift = np.random.randint(-5, 7)
            notes[:, 2] = np.clip(notes[:, 2] + pitch_shift, 0, 127)

        # Randomly select a starting beat
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
            tag_path = self.tag_dir / name[2] / name[3]/f"{name}.txt"
            # print()
            # tag = np.load(self.tag_dir / name[2] / name[3]/f"{name}.txt", allow_pickle=True)
            with open(tag_path, "r") as file:
                tag = file.readlines()
            tag_list = [i.strip() for i in tag]
            # print(f"tags is = {tag_list}")
            seq = self.encode_fn(notes, self.encoding, self.indexer,self.tags,tag_list)
            # Encode the notes
            ## notes, encoding, indexer,tag_names,tag: tag can be a list, self.tags: name of tag
            if self.max_seq_len is not None and len(seq) > self.max_seq_len:
                seq = np.concatenate((seq[: self.max_seq_len - 2], seq[-2:]))
        if self.tags == None:
            ## No tag, tag=None
            seq = self.encode_fn(notes, self.encoding, self.indexer,self.tags, None)
            if self.max_seq_len is not None and len(seq) > self.max_seq_len:
                seq = np.concatenate((seq[: self.max_seq_len - 2], seq[-2:]))
        return {"name": name, "seq": seq}

    @classmethod
    def collate(cls, data):

        seq = [sample["seq"] for sample in data]
        return {
            "name": [sample["name"] for sample in data],
            "seq": torch.tensor(pad(seq), dtype=torch.long),
            "seq_len": torch.tensor([len(s) for s in seq], dtype=torch.long),
            "mask": get_mask(seq),
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
    dataset = MusicDataset(
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
    data_loader = torch.utils.data.DataLoader(
        dataset, args.batch_size, True, collate_fn=MusicDataset.collate
    )

    # Iterate over the loader
    n_batches = 0
    n_samples = 0
    seq_lens = []
    # first element in a dataloader
    # print(f"first sequence in a dataloader{next(iter(data_loader))['seq']}")
    # for name, seq in data_loader:
    #     print(f"name = {name}")
    for i, batch in enumerate(data_loader):
        # print(f"batch = {batch}")
        n_batches += 1
        # if batch == None:
        #     continue
        n_samples += len(batch["name"])

        seq_lens.extend(int(l) for l in batch["seq_len"])
        if i == 0:
            logging.info("Example:")
            for key, value in batch.items():
                if key == "name":
                    continue
                logging.info(f"Shape of {key}: {value.shape}")
            logging.info(f"Name: {batch['name'][0]}")
        
    logging.info(
        f"Successfully loaded {n_batches} batches ({n_samples} samples)."
    )

    # Print sequence length statistics
    logging.info(f"Avg sequence length: {np.mean(seq_lens):2f}")
    logging.info(f"Min sequence length: {min(seq_lens)}")
    logging.info(f"Max sequence length: {max(seq_lens)}")


if __name__ == "__main__":

    #         if keyword not in keywordList:
    #             keywordList.append(keyword)
    # pprint(keywordList)
    main()
