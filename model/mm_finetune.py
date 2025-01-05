import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig, AutoConfig
from accelerate import PartialState
import wandb
import torch.optim as optim
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
import huggingface_hub
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
from transformers import AutoModel
import sys
import transformers
import os
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

# Specify the model name
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    # removed negative pitch file
    parser.add_argument("-nt", "--train_names", type=str, help="train names", default=Path("/data2/weihanx/musicgpt/mm_ft_file/train-names_cleaned.txt"))
    parser.add_argument("-nv", "--valid_names", type=str, help="valid names", default=Path("/data2/weihanx/musicgpt/mm_ft_file/valid-names_cleaned.txt"))   
    parser.add_argument(
        "-i", "--in_dir", type=pathlib.Path, help="note data directory", default=Path("/data2/weihanx/musicgpt/combined_dataset/musescore_notes")
    )
    parser.add_argument(
        "-t", "--text_dir", type=pathlib.Path, help="text data directory", default=Path("/data2/weihanx/musicgpt/metascore_plus/metascore_plus_tag_without_genre_cap")
    )
    parser.add_argument(
        "-o", "--out_dir", type=pathlib.Path, help="output directory", default=Path("/data2/weihanx/musicgpt/bloom_ft_model")
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


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device_map="DDP" # for DDP and running with `accelerate launch test_sft.py`

# if device_map == "DDP":
#     device_string = PartialState().process_index
#     device_map={'':device_string}
# print(f"device_map = {device_map}")
base_model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m")
args = parse_args()

tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m") # text tokenizer
embedding_size = base_model.config.hidden_size

# Get the tokenizer type
tokenizer_type = type(tokenizer).__name__

print(f"Embedding Size: {embedding_size}") # 4096
print(f"Tokenizer Type: {tokenizer_type}") # PreTrainedTokenizerFast

if args.representation == "remi":
    representation = representation_remi
    encoding = representation_remi.load_encoding("/data2/weihanx/musicgpt/encoding_remi.json")

rep_dict = representation.load_encoding("/data2/weihanx/musicgpt/encoding_remi.json")

new_vocab = list(rep_dict["event_code_map"].keys())
num_new_word = len(list(rep_dict["event_code_map"].keys()))

current_embeddings = base_model.get_input_embeddings()

average_embedding = torch.mean(current_embeddings.weight.data, dim=0)

# Get the current number of tokens and the embedding dimension
old_num_vocab, embedding_dim = current_embeddings.weight.size() # old vocab size * embedding size

# old_voca_size = base_model.config.vocab_size # same as number of token in tokenizer # 250880
# tokenizer_vocab_size = tokenizer.vocab_size # 250680
# print(f"old num of class = {old_voca_size} cur_num_vocab = {tokenizer_vocab_size}") # 250880, 250680
# Create a new embedding layer with additional ocab}"

new_vocab_size = 252416 # 250880 + 4*128*3 --> should be multiple of 4*128 for parallel

base_model.config.vocab_size = new_vocab_size

new_embeddings = torch.nn.Embedding(new_vocab_size, embedding_dim)

# 250880: word embedding
new_embeddings.weight.data[:old_num_vocab] = current_embeddings.weight.data #  Initialize the new embeddings with the existing weights 

# initilzie to the the token name before underscore
# Initialize new tokens with embeddings of their base tokens
for new_token in new_vocab:
    base_token = new_token.split('_')[0]  # Extract the base token
    base_token_id = tokenizer.convert_tokens_to_ids(base_token)  # Get the token ID of the base token
    new_token_id = old_num_vocab + new_vocab.index(new_token)  # vocab size + index of the new token

    # Check if the base token exists in the current vocabulary
    if base_token_id is not None:
        new_embeddings.weight.data[new_token_id] = current_embeddings.weight.data[base_token_id]
    else:
        # If the base token does not exist, initialize with random weights
        new_embeddings.weight.data[new_token_id] = average_embedding

# Update the model's embedding layer
# original: no dummy token, should always less than 250680
base_model.resize_token_embeddings(new_vocab_size) # extend the vocab when getting loss, if not add, new token will be outsize of class
base_model.set_input_embeddings(new_embeddings)
# update the embedding
current_embeddings = base_model.get_input_embeddings()

# Get the current number of tokens and the embedding dimension
num_tokens, embedding_dim = current_embeddings.weight.size()

# print(f"Undated vocab size = {base_model.config.vocab_size}, check: 252190")


train_dataset = MusicDataset(
    args.train_names,
    args.in_dir,
    args.text_dir,
    encoding=encoding,
    max_seq_len=args.max_seq_len,
    max_beat=args.max_beat,
    use_csv=args.use_csv,
    use_augmentation=args.aug,

)

val_dataset = MusicDataset(
    args.valid_names,
    args.in_dir,
    args.text_dir,
    encoding=encoding,
    max_seq_len=args.max_seq_len,
    max_beat=args.max_beat,
    use_csv=args.use_csv,
    use_augmentation=False,
)
print(f"total training samples = {len(train_dataset)}. Total validation samples = {len(val_dataset)}")

from transformers import TrainerCallback

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.epochs_no_improve = 0

    def on_evaluate(self, args, state, control, **kwargs):
        eval_loss = kwargs['metrics']['eval_loss']
        
        if self.best_loss is None or (self.best_loss - eval_loss) > self.min_delta:
            self.best_loss = eval_loss
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1
        
        if self.epochs_no_improve >= self.patience:
            print(f"Early stopping triggered. No improvement in {self.patience} epochs.")
            control.should_training_stop = True
#---------Training Arguments----------#
wandb.init(project="mm_ft_1220")

model = base_model
print(f"model.device = {device}")
model = model.to(device)
model.is_parallelizable = True
model.model_parallel = True

# data loader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8, collate_fn=MusicDataset.collate)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, collate_fn=MusicDataset.collate)
n_parameters = sum(p.numel() for p in model.parameters())
n_trainables = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)
# directly finetune
lora_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.1,
    r=32,
    bias="lora_only",
    # out.txt has all name, selected only last a few layers and emebdding layers
    target_modules = [
    "word_embeddings",
    "query_key_value",
    "dense",
    "dense_h_to_4h",
    "dense_4h_to_h",
    "lm_head"
],
    task_type="CAUSAL_LM",
)

### directly finetune without lora
model = get_peft_model(base_model, lora_config)
optimizer = optim.AdamW(model.parameters(), lr=1e-5) 
# print(f"model.config = {model.config}")
model = model.to(device)
# model.config.device_map = "DDP"
n_parameters = sum(p.numel() for p in model.parameters())

n_trainables = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)

# trainable layer in this set
# Assuming 'model' is your PyTorch model
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Layer {name} is trainable.")
    else:
        print(f"Layer {name} is not trainable.")
num_epochs = 10
val_loss = 0.0

for i in range(num_epochs):
    print(f"Epoch {i}")
    model.train()  # Set the model to training mode
    train_loss = 0.0  # Initialize train loss for this epoch

    for batch in train_loader:
        inputs = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]
        # put onto decice
        inputs = inputs.to(device)
        labels = labels.to(device)
        attention_mask = attention_mask.to(device)
        # Displaying batch contents can be verbose; consider limiting such prints
        # print(f"input = {inputs[0][:50]}")
        # print(f"labels = {labels[0][:50]}")
        # print(f"attention_mask = {attention_mask[0][:50]}")

        optimizer.zero_grad()
        outputs = model(input_ids=inputs, labels=labels, attention_mask=attention_mask)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()  # Accumulate the loss

    # Log and print average training loss
    average_train_loss = train_loss / len(train_loader)
    wandb.log({"train_loss": average_train_loss})
    print(f"Average Training Loss: {average_train_loss}")

    # Evaluation phase
    model.eval()
    val_loss = 0.0  # Reset validation loss for the epoch
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["input_ids"]
            labels = batch["labels"]
            attention_mask = batch["attention_mask"]
            # put onto decice
            inputs = inputs.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(input_ids=inputs, labels=labels, attention_mask=attention_mask)
            val_loss += outputs.loss.item()  # Sum up batch loss

        average_val_loss = val_loss / len(val_loader)  # Calculate average validation loss
        wandb.log({"val_loss": average_val_loss})
        print(f"Average Validation Loss: {average_val_loss}")
    if i % 2 == 0:
# save model
        model.save_pretrained(f"/data2/weihanx/musicgpt/bloom_ft_1220_{i}")



# training_args = TrainingArguments(
#     output_dir='./mm_ft_1207',
#     evaluation_strategy='epoch',
#     learning_rate=1e-4, # 1e-4
#     per_device_train_batch_size=4, # 4 is max
#     per_device_eval_batch_size=1,
#     num_train_epochs=10,
#     warmup_steps=500,
#     lr_scheduler_type='cosine',
#     weight_decay=0.01,
#     save_total_limit=10,
#     gradient_accumulation_steps=2,
#     dataloader_num_workers=8,
#     fp16=True,
#     save_steps=10000,  # Save every epochs
#     logging_steps=500,  # Adjust based on your preference
#     report_to="wandb",  # Enable logging to wandb
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset, 
#     eval_dataset=val_dataset,
#     data_collator=MusicDataset.collate,
#     callbacks=[EarlyStoppingCallback(patience=3, min_delta=0.01)]
# )

# # # trainer.train()

# # # # # Train
# # trainer.train()
# # model.save_pretrained("/data2/weihanx/musicgpt/bloom_ft_1207")
# # wandb.finish()