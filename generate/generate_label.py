# This file is used to generate genre labels based on MMT classfication
"""
Steps:
1. load 20 files from balanced dataset
2. load 20 files from files without genre labels
"""

import argparse
import logging
import pathlib
import pprint
import shutil
import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from x_transformers import XTransformer
import dataset_mmt
import representation_program_drum
import linear_attention_transformer
import utils
import torch.optim as optim
## Linear Transformer with Classification Output
import torch
import torch.nn as nn
from linear_attention_transformer import LinearAttentionTransformerLM
from sklearn.metrics import precision_recall_curve
from x_transformers.autoregressive_wrapper import (
    ENTMAX_ALPHA,
    entmax,
    exists,
    top_a,
    top_k,
    top_p,
)
from x_transformers.x_transformers import (
    AbsolutePositionalEmbedding,
    AttentionLayers,
    Encoder, # using decoder in x_transformers
    TokenEmbedding,
    always,
    default,
    exists,
)
@utils.resolve_paths
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        choices=("sod", "lmd", "lmd_full", "muse"),
        required=False,
        help="dataset key",
    )
    parser.add_argument(
        "-r",
        "--representation",
        choices=("compact", "mmm", "remi"),
        default="remi",
        help="representation key",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=int,
        required = True,
        help="model epoch",
    )
    parser.add_argument(
        "-t", "--test_names", type=pathlib.Path, help="training names"
    )
    parser.add_argument(
        "-i", "--in_dir", type=pathlib.Path, help="input data directory"
    )

    parser.add_argument(
        "-o", "--out_dir", type=pathlib.Path, help="output directory"
    )
    parser.add_argument(
        "-it", "--in_dir_tag", type=pathlib.Path, help="input tag directory"
    )
    # Data
    parser.add_argument(
        "-bs",
        "--batch_size",
        default=32,
        type=int,
        help="batch size",
    )
    parser.add_argument(
        "--grad_acc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to accumulate gradients to increase the batch size",
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
    # Model
    parser.add_argument(
        "--max_seq_len",
        default=3072,
        type=int,
        help="maximum sequence length",
    )
    parser.add_argument(
        "--max_beat",
        default=64,
        type=int,
        help="maximum number of beats",
    )
    parser.add_argument("--dim", default=768, type=int, help="model dimension")
    parser.add_argument(
        "-l", "--layers", default=12, type=int, help="number of layers"
    )
    parser.add_argument(
        "--heads", default=12, type=int, help="number of attention heads"
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="dropout rate"
    )
    parser.add_argument(
        "--abs_pos_emb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to use absolute positional embedding",
    )
    parser.add_argument(
        "--rel_pos_emb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="whether to use relative positional embedding",
    )
    # Training
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
    # Others
    parser.add_argument(
        "-g", "--gpu", nargs="+", type=int, help="gpu number(s)"
    )
    parser.add_argument(
        "-j",
        "--jobs",
        default=4,
        type=int,
        help="number of workers for data loading",
    )
    parser.add_argument(
        "-q", "--quiet", action="count", default=0, help="show warnings only"
    )
    return parser.parse_args(args=args, namespace=namespace)


class MusicTransformerWrapper(nn.Module):
    def __init__(
        self,
        *,
        encoding,
        max_seq_len,
        attn_layers,
        emb_dim=None,
        max_beat=None,
        max_mem_len=0.0,
        shift_mem_down=0,
        emb_dropout=0.0,
        num_memory_tokens=None,
        tie_embedding=False,
        use_abs_pos_emb=True,
        l2norm_embed=False,
    ):
        super().__init__()
        assert isinstance(
            attn_layers, AttentionLayers
        ), "attention layers must be one of Encoder or Decoder"

        dim = attn_layers.dim
        emb_dim = default(emb_dim, dim)

        self.max_seq_len = max_seq_len
        self.max_mem_len = max_mem_len
        self.shift_mem_down = shift_mem_down

        n_tokens = encoding["n_tokens"]
        if max_beat is not None:
            beat_dim = encoding["dimensions"].index("beat")
            n_tokens[beat_dim] = max_beat + 1

        self.l2norm_embed = l2norm_embed
        self.token_emb = nn.ModuleList(
            [
                TokenEmbedding(emb_dim, n, l2norm_embed=l2norm_embed)
                for n in n_tokens
            ]
        )
        self.pos_emb = (
            AbsolutePositionalEmbedding(
                emb_dim, max_seq_len, l2norm_embed=l2norm_embed
            )
            if (use_abs_pos_emb and not attn_layers.has_pos_emb)
            else always(0)
        )

        self.emb_dropout = nn.Dropout(emb_dropout)

        self.project_emb = (
            nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()
        )
        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)

        self.init_()

        self.to_logits = (
            nn.ModuleList([nn.Linear(dim, n) for n in n_tokens])
            if not tie_embedding
            else [lambda t: t @ emb.weight.t() for emb in self.token_emb]
        )

        # memory tokens (like [cls]) from Memory Transformers paper
        num_memory_tokens = default(num_memory_tokens, 0)
        self.num_memory_tokens = num_memory_tokens
        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(
                torch.randn(num_memory_tokens, dim)
            )

    def init_(self):
        if self.l2norm_embed:
            for emb in self.token_emb:
                nn.init.normal_(emb.emb.weight, std=1e-5)
            nn.init.normal_(self.pos_emb.emb.weight, std=1e-5)
            return

        for emb in self.token_emb:
            nn.init.kaiming_normal_(emb.emb.weight)

    def forward(
        self,
        x,  # shape : (b, n , d)
        return_embeddings=False,
        mask=None,
        return_mems=False,
        return_attn=False,
        mems=None,
        **kwargs,
    ):
        b, _, _ = x.shape
        num_mem = self.num_memory_tokens

        x = sum(
            emb(x[..., i]) for i, emb in enumerate(self.token_emb)
        ) + self.pos_emb(x)
        x = self.emb_dropout(x)

        x = self.project_emb(x)

        if num_mem > 0:
            mem = repeat(self.memory_tokens, "n d -> b n d", b=b)
            x = torch.cat((mem, x), dim=1)

            # auto-handle masking after appending memory tokens
            if exists(mask):
                mask = F.pad(mask, (num_mem, 0), value=True)

        if self.shift_mem_down and exists(mems):
            mems_l, mems_r = (
                mems[: self.shift_mem_down],
                mems[self.shift_mem_down :],
            )
            mems = [*mems_r, *mems_l]
        # bidiretional, this mask is a padding mask, I have another attention mask is None
        x, intermediates = self.attn_layers(
            x, mask=mask, mems=mems, return_hiddens=True, **kwargs
        )
        x = self.norm(x)

        mem, x = x[:, :num_mem], x[:, num_mem:]

        out = (
            [to_logit(x) for to_logit in self.to_logits]
            if not return_embeddings
            else x
        )

        if return_mems:
            hiddens = intermediates.hiddens
            new_mems = (
                list(
                    map(
                        lambda pair: torch.cat(pair, dim=-2),
                        zip(mems, hiddens),
                    )
                )
                if exists(mems)
                else hiddens
            )
            new_mems = list(
                map(
                    lambda t: t[..., -self.max_mem_len :, :].detach(), new_mems
                )
            )
            return out, new_mems

        if return_attn:
            attn_maps = list(
                map(
                    lambda t: t.post_softmax_attn,
                    intermediates.attn_intermediates,
                )
            )
            return out, attn_maps

        return out
        
# change to multi-label classification
class MusicXTransformer(nn.Module):
    def __init__(self, *, dim, encoding, **kwargs):
        """
        *: It indicates that all following parameters (until another asterisk is encountered) must be specified as keyword arguments. 
        **kwargs: It's a common practice to use **kwargs to accept arbitrary keyword arguments. dictionary type
                    This is particularly useful for extending classes or methods,
                    as any additional keyword arguments can be passed through **kwargs 
                    without needing to modify the function signature.
        """
        super().__init__()
        assert "dim" not in kwargs, "dimension must be set with `dim` keyword"
        ## remove the key pair after retrive information
        transformer_kwargs = {
            "max_seq_len": kwargs.pop("max_seq_len"),
            "max_beat": kwargs.pop("max_beat"),
            "emb_dropout": kwargs.pop("emb_dropout", 0),
            "use_abs_pos_emb": kwargs.pop("use_abs_pos_emb", True),
        }
        # bert like
        self.encoder = MusicTransformerWrapper(
            encoding=encoding,
            attn_layers=Encoder(dim=dim, **kwargs),
            **transformer_kwargs,
        )
    def forward(self, seq, mask=None, **kwargs):
        return self.encoder(seq, mask=mask, **kwargs)



class TransformerForClassification(nn.Module):
    def __init__(self, encoding, num_classes, args):
        super().__init__()
        self.transformer = MusicXTransformer(
        dim=128,
        encoding=encoding,
        depth=3,
        heads=4,
        max_seq_len=1024,
        max_beat=256,
        rel_pos_bias=True,  # relative positional bias
        rotary_pos_emb=True,  # rotary positional encoding
        emb_dropout=0.1,
        attn_dropout=0.1,
        ff_dropout=0.1,
    )
        self.classifier = nn.Linear(514, num_classes) 
    def forward(self, x, mask):
        # print(f"input shape = {x.shape}") # list for 6 elements [32, 1024, dim]
        x = self.transformer(x, mask=mask)
        # torch.Size([32, 1024, 5]),torch.Size([32, 1024, 257]), torch.Size([32, 1024, 25]), torch.Size([32, 1024, 129]), torch.Size([32, 1024, 33]),torch.Size([32, 1024, 65])
        # print(f"shape of mmt output = {x[0].shape},{x[1].shape}, {x[2].shape}, {x[3].shape}, {x[4].shape},{x[5].shape}")
        # x_1, x_2, x_3, x_4, x_5, x_6 = x
        # selected_tensors = [tensor[:, 0, :] for tensor in [tensor1, tensor2, tensor3, tensor4, tensor5, tensor6]]
        x_out = [out[:,0,:] for out in x]
        x_out = torch.cat(x_out, dim=1)
        # x = x[:, 0, :]  # [batch_size, dim]
        return self.classifier(x_out)

# Example instantiation

args = parse_args()

# Set up the logger
logging.basicConfig(
    level=logging.DEBUG + 10 * args.quiet,
    format="%(message)s",
    handlers=[
        logging.FileHandler(args.out_dir / "genre_classifier_mmt_labels.log", "w"),
        logging.StreamHandler(sys.stdout),
    ],
)

# Log command called
logging.info(f"Running command: python {' '.join(sys.argv)}")

# Log arguments
logging.info(f"Using arguments:\n{pprint.pformat(vars(args))}")

# Save command-line arguments
logging.info(f"Saved arguments to {args.out_dir / 'test-args.json'}")

utils.save_args(args.out_dir / "test-args.json", args)

device = torch.device(
    f"cuda:{args.gpu[0]}" if args.gpu is not None else "cpu"
)

print(f"device = {device}")


dataset = dataset_mmt # output: return {"name": name, "seq": seq, "label": tag_map[tag_list[0]]}



representation = representation_program_mmt # [SOS] NOTES [EOS]
# Load the indexer
encoding = representation_program_mmt.load_encoding(args.in_dir / "encoding_mmt.json")

# should be a pair of music and label
# representation --> dataset, need to change dataset
model = TransformerForClassification(
    encoding = encoding,
    num_classes=17,  # Replace with the actual number of classes: I have 17 genre class, remove voice
    args=args
)


model_name = f"model_{args.model}.pt"


path = args.out_dir / "checkpoints" / model_name

model.load_state_dict(torch.load(path))

criterion = torch.nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Create the optimizer
    # for regulariztiaon
optimizer = torch.optim.AdamW(
    model.parameters(), args.learning_rate, weight_decay=args.weight_decay
)

logging.info(f"load model: {model_name} from {path}")
# Set the model to evaluation mode if you are doing inference/testing
model = model.to(device)
model.eval()

test_dataset = dataset.MusicDataset(
    args.test_names,
    args.in_dir,
    args.in_dir_tag,
    encoding=encoding,
    max_seq_len=args.max_seq_len,
    max_beat=args.max_beat,
    use_csv=args.use_csv,
    use_augmentation=args.aug,
)



logging.info(f"Using device: {device}")
logging.debug(f"Using batch size: {args.batch_size}")

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    args.batch_size,
    num_workers=args.jobs,
    collate_fn=dataset.MusicDataset.collate,
)

model.eval() # stop training
num_classes = 17
"""
only for test
with torch.no_grad():
    all_labels = []
    all_outputs = []
    total_loss = 0
    count = 0
    correct = 0
    total = 0
    for batch in test_loader:
        seq = batch['seq'].to(device) 
        label = batch['label']
        outputs = model(seq)
        label_tensor = torch.tensor(label, dtype=torch.float,  device = device)
        loss = criterion(outputs, label_tensor)
        all_outputs.append(outputs.cpu())
        all_labels.append(label_tensor.cpu())
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    probabilities = torch.sigmoid(all_outputs)
    optimal_thresholds = []
    num_classes = probabilities.shape[1]
    for i in range(num_classes):
        # Calculate precision and recall for various thresholds
        precision, recall, thresholds = precision_recall_curve(all_labels[:, i].numpy(), probabilities[:, i].numpy())
        # Calculate F1 score for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-12)  # Adding a small constant to avoid division by zero
        # Find the threshold where F1 score is maximized
        optimal_idx = np.argmax(f1_scores)
        logging.info(f"Test for epoch = {20} class = {i}, optimal f1_score = {f1_scores[optimal_idx]}, corresponding threshold = {thresholds[optimal_idx]}")
        optimal_thresholds.append(thresholds[optimal_idx])
    # Print or log the optimal thresholds for each class
    logging.info(f"Optimal thresholds per class:{optimal_thresholds}")
"""

def get_tags_for_tensor(predictions, index_to_tag):
    tags = []
    for row in predictions:
        row_tags = [index_to_tag[i] for i, val in enumerate(row) if val]
        tags.append(row_tags)
    return tags


with torch.no_grad():
    total_loss = 0
    count = 0

    all_labels = []
    all_outputs = []
    # epoch 2
    # 12/18 classifier_exp_half_program
    thresholds = [0.7736253, 0.3684758, 0.008227954, 0.28102994, 0.6925298, 0.05307746, 0.014298081, 0.05175175, 0.035709597, 0.16332532, 2.9449953e-14, 0.12994479, 0.14131203, 6.413033e-10, 2.7628653e-14, 0.019064851, 2.9681685e-05]
    for batch in test_loader: # only keep one batch: 32
        name = batch['name']
        logging.info(f"Song Name = {name}")
        seq = batch['seq'].to(device)
        label = batch["label"]
        # labels = batch['label'].to(device)  # Multi-label, so multiple labels per instance, notag don't  have underground label
        mask = batch['mask'].to(device)
        # Pass through the model
        outputs = model(seq, mask=mask)
        label_tensor = torch.tensor(label, dtype=torch.float,  device = device)
        # loss = criterion(outputs, label_tensor)
        all_outputs.append(outputs)
        all_labels.append(label_tensor)
        # Compare outputs with thresholds
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    probabilities = torch.sigmoid(all_outputs)

    tag_map = {
    'classical': 0, 'soundtrack': 1, 'religious music': 2, 'folk': 3, 'pop_rock': 4, 
    'hip hop': 5, 'metal': 6, 'world music': 7, 'r&b, funk & soul': 8, 
    'electronic': 9, 'reggae & ska': 10, 'country': 11, 'jazz': 12, 'comedy': 13, 
    'blues': 14, 'new age': 15, 'disco': 16
    }

# # Inverting the tag map to map indices to tag names
    index_to_tag = {v: k for k, v in tag_map.items()}

    print(f"label = {label_tensor}")
    # print(f"outputs = {outputs}")

    predictions = probabilities > torch.tensor(thresholds).to(device)
    # convert prediction tensors to tag list
    # convert label_tensor to tag list
# Getting tags for each row in the tensor
    predictions_tags = get_tags_for_tensor(predictions, index_to_tag)
    label_tags = get_tags_for_tensor(label_tensor, index_to_tag)
    # logging.info(f"Predictions = {predictions_tags} with No Label")
    logging.info(f"Predictions = {predictions_tags} with Label being = {label_tags}")   
    # genreate 