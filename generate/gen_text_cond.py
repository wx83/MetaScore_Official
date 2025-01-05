import argparse
import logging
import pathlib
import pprint
import subprocess
import sys

# import linear_attention_transformer_modified 
# from linear_attention_transformer_modified import linear_attention_transformer
import linear_attention_transformer
import matplotlib.pyplot as plt
import numpy as np
import performer_pytorch
import torch
import torch.utils.data
import tqdm
import x_transformers

import music_x_transformers
import dataset_text_cond
import representation_remi
import utils
import muspy 
import numpy as np
import torch
import torch.utils.data
import tqdm
import x_transformers

### TODO: CHANGE REPRESENTATION, change dataset tag
### Caution: truth is a simple music sheet, test name should no prefix
import utils

from collections import defaultdict
@utils.resolve_paths
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--names", type=pathlib.Path, help="input names")
    parser.add_argument(
        "-i", "--in_dir", type=pathlib.Path, help="input data directory"
    )
    parser.add_argument(
        "-o", "--out_dir", type=pathlib.Path, help="output directory"
    )
    parser.add_argument(
        "-ns",
        "--n_samples",
        default=50,
        type=int,
        help="number of samples to generate",
    )
    # Data
    parser.add_argument(
        "-s",
        "--shuffle",
        action="store_true",
        help="whether to shuffle the test data",
    )
    parser.add_argument(
        "--use_csv",
        action="store_true",
        help="whether to save outputs in CSV format (default to NPY format)",
    )
    # Model: can change to not loading the best model
    parser.add_argument(
        "--model_steps",
        type=int,
        help="step of the trained model to load (default to the best model)",
    )
    parser.add_argument(
        "-it", "--in_dir_tag", type=pathlib.Path, help="input tag directory"
    )
    # decide the result leng
    parser.add_argument(
        "--seq_len",
        type=int,
        required = True,
        help="sequence length to generate (default to max sequence length)",
    )
    parser.add_argument(
        "--temperature",
        nargs="+",
        default=1.0,
        type=float,
        help="sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--filter",
        nargs="+",
        default="top_k",
        type=str,
        help="sampling filter (default: 'top_k')",
    )
    parser.add_argument(
        "--filter_threshold",
        nargs="+",
        default=0.9,
        type=float,
        help="sampling filter threshold (default: 0.9)",
    )
    # Others
    parser.add_argument("-g", "--gpu", type=int, help="gpu number")
    parser.add_argument(
        "-j", "--jobs", default=1, type=int, help="number of jobs"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="show warnings only"
    )
    return parser.parse_args(args=args, namespace=namespace)


def save_pianoroll(filename, music, size=None, **kwargs):
    """Save the piano roll to file."""
    music.show_pianoroll(track_label="program", **kwargs)
    if size is not None:
        plt.gcf().set_size_inches(size)
    plt.savefig(filename)
    plt.close()


def save_result(
    filename, data, sample_dir, encoding, vocabulary, representation
):
    """Save the results in multiple formats."""
    # Save as a numpy array
    np.save(sample_dir / "npy" / f"{filename}.npy", data)

    # Save as a CSV file
    representation.save_csv_codes(sample_dir / "csv" / f"{filename}.csv", data)

    # Save as a TXT file
    representation.save_txt(
        sample_dir / "txt" / f"{filename}.txt", data, vocabulary
    )

    # Convert to a MusPy Music object
    music = representation.decode(data, encoding, vocabulary)
    m2 = music.copy 
    # m2.set_resolution(4), save m2 new midi
    # Save as a MusPy JSON file
    music.save(sample_dir / "json" / f"{filename}.json")

    # Save as a piano roll
    # save_pianoroll(
    #     sample_dir / "png" / f"{filename}.png", music, (20, 5), preset="frame"
    # )

    # Save as a MIDI file
    music.write(sample_dir / "mid" / f"{filename}.mid")
    # lower resolution,
    # Save as a WAV file
    music.write(
        sample_dir / "wav" / f"{filename}.wav",
        options="-o synth.polyphony=4096",
    )

    # Save also as a MP3 file
    # subprocess.check_output(
    #     ["ffmpeg", "-loglevel", "error", "-y", "-i"]
    #     + [str(sample_dir / "wav" / f"{filename}.wav")]
    #     + ["-b:a", "192k"]
    #     + [str(sample_dir / "mp3" / f"{filename}.mp3")]
    # )

    # Trim the music
    music.trim(music.resolution * 64)

    # # Save the trimmed version as a piano roll
    # save_pianoroll(
    #     sample_dir / "png-trimmed" / f"{filename}.png", music, (10, 5)
    # )

    # Save as a WAV file
    music.write(
        sample_dir / "wav-trimmed" / f"{filename}.wav",
        options="-o synth.polyphony=4096",
    )

    # Save also as a MP3 file
    subprocess.check_output(
        ["ffmpeg", "-loglevel", "error", "-y", "-i"]
        + [str(sample_dir / "wav-trimmed" / f"{filename}.wav")]
        + ["-b:a", "192k"]
        + [str(sample_dir / "mp3-trimmed" / f"{filename}.mp3")]
    )


def evaluate(data, encoding, vocabulary, representation, filename, eval_dir):
    """Evaluate the results."""
    # Save as a numpy array
    np.save(eval_dir / "npy" / f"{filename}.npy", data)

    # Save as a CSV file
    representation.save_csv_codes(eval_dir / "csv" / f"{filename}.csv", data)

    # Convert to a MusPy Music object
    music = representation.decode(data, encoding, vocabulary)
    music.tracks = [t for t in music if not t.is_drum]
    # Save as a MusPy JSON file
    music.save(eval_dir / "json" / f"{filename}.json")

    if not music.tracks:
        return {
            "pitch_class_entropy": np.nan,
            "scale_consistency": np.nan,
            "groove_consistency": np.nan,
        }

    return {
        "pitch_class_entropy": muspy.pitch_class_entropy(music),
        "scale_consistency": muspy.scale_consistency(music),
        "groove_consistency": muspy.groove_consistency(
            music, 4 * music.resolution
        ),
    }


def main():
    """Main function."""
    # Parse the command-line arguments
    args = parse_args()

    # # Set default arguments
    # if args.dataset is not None:
    #     if args.names is None:
    #         args.names = pathlib.Path(
    #             f"data/{args.dataset}/processed/test-names.txt"
    #         )
    #     if args.in_dir is None:
    #         args.in_dir = pathlib.Path(f"data/{args.dataset}/processed/notes/")
    #     if args.out_dir is None:
    #         args.out_dir = pathlib.Path(f"exp/test_{args.dataset}")

    # Set up the logger
    logging.basicConfig(
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(args.out_dir / "generate_text_cond.log", "w"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Log command called
    logging.info(f"Running command: python {' '.join(sys.argv)}")

    # Log arguments
    logging.info(f"Using arguments:\n{pprint.pformat(vars(args))}")

    # Save command-line arguments
    logging.info(f"Saved arguments to {args.out_dir / 'generate-args.json'}")
    utils.save_args(args.out_dir / "generate-args.json", args)

    # Load training configurations
    logging.info(
        f"Loading training arguments from: {args.out_dir / 'train-args.json'}"
    )
    train_args = utils.load_json(args.out_dir / "train-args.json")
    logging.info(f"Using loaded arguments:\n{pprint.pformat(train_args)}")
    eval_dir = args.out_dir / "eval"
    eval_dir.mkdir(exist_ok=True)
    for key in ("truth", "conditioned"):
        (eval_dir / key).mkdir(exist_ok=True)
        (eval_dir / key / "npy").mkdir(exist_ok=True)
        (eval_dir / key / "csv").mkdir(exist_ok=True)
        (eval_dir / key / "json").mkdir(exist_ok=True)

    # Set variables

    representation = representation_remi
    dataset = dataset_text_cond
    sentence_name = "highrate"
    # Make sure the sample directory exists ## Change here to distinguish different generated samples
    sample_dir = args.out_dir / f"textcond_1022_{args.model_steps}_{sentence_name}"
    sample_dir.mkdir(exist_ok=True)
    (sample_dir / "npy").mkdir(exist_ok=True)
    (sample_dir / "csv").mkdir(exist_ok=True)
    (sample_dir / "txt").mkdir(exist_ok=True)
    (sample_dir / "json").mkdir(exist_ok=True)
    (sample_dir / "png").mkdir(exist_ok=True)
    (sample_dir / "mid").mkdir(exist_ok=True)
    (sample_dir / "wav").mkdir(exist_ok=True)
    (sample_dir / "mp3").mkdir(exist_ok=True)
    (sample_dir / "png-trimmed").mkdir(exist_ok=True)
    (sample_dir / "wav-trimmed").mkdir(exist_ok=True)
    (sample_dir / "mp3-trimmed").mkdir(exist_ok=True)

    # Get the specified device
    device = torch.device(
        f"cuda:{args.gpu}" if args.gpu is not None else "cpu"
    )
    logging.info(f"Using device: {device}")

    # Get representation
    # if train_args["representation"] == "mmm":
    #     representation = representation_mmm
    # elif train_args["representation"] == "remi":
    #     representation = representation_genre
    # else:
    #     raise ValueError(
    #         f"Unknown representation: {train_args['representation']}"
    #     )

    # Load the encoding
    encoding = representation.get_encoding()
    # Load the indexer
    indexer = representation.Indexer(encoding["event_code_map"])
    # print(f"indexer = {encoding['eventx_code_map']}")
    # Get the vocabulary
    vocabulary = encoding["code_event_map"]

    # Create the dataset and data loader
    logging.info(f"Creating the data loader...")
    test_dataset = dataset.MusicDataset(
        args.names,
        args.in_dir,
        encoding=encoding,
        indexer=indexer,
        encode_fn=representation.encode_notes,
        max_seq_len=train_args["max_seq_len"],
        max_beat=train_args["max_beat"],
        use_csv=args.use_csv,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=args.shuffle,
        num_workers=args.jobs,
        collate_fn=dataset.MusicDataset.collate,
    )

    # Create the model
    logging.debug(f"Creating model...")
    # if train_args["model"] == "compact":
    #     model = music_x_transformers.MusicTransformerWrapper(
    #         encoding=encoding,
    #         max_seq_len=train_args["max_seq_len"],
    #         max_beat=train_args["max_beat"],
    #         use_abs_pos_emb=train_args["abs_pos_emb"],
    #         emb_dropout=train_args["dropout"],
    #         attn_layers=music_x_transformers.Decoder(
    #             dim=train_args["dim"],
    #             depth=train_args["layers"],
    #             heads=train_args["heads"],
    #             rotary_pos_emb=train_args["rel_pos_emb"],
    #             attn_dropout=train_args["dropout"],
    #             ff_dropout=train_args["dropout"],
    #         ),
    #     )
    #     model = music_x_transformers.MusicAutoregressiveWrapper(
    #         model, encoding=encoding
    #     )
    # elif train_args["model"] == "transformer":
    #     model = x_transformers.TransformerWrapper(
    #         num_tokens=len(indexer),
    #         max_seq_len=train_args["max_seq_len"],
    #         use_abs_pos_emb=train_args["abs_pos_emb"],
    #         emb_dropout=train_args["dropout"],
    #         attn_layers=x_transformers.Decoder(
    #             dim=train_args["dim"],
    #             depth=train_args["layers"],
    #             heads=train_args["heads"],
    #             rotary_pos_emb=train_args["rel_pos_emb"],
    #             attn_dropout=train_args["dropout"],
    #             ff_dropout=train_args["dropout"],
    #         ),
    #     )
    #     model = x_transformers.AutoregressiveWrapper(model)
    # elif train_args["model"] == "performer":
    #     model = performer_pytorch.PerformerLM(
    #         num_tokens=len(indexer),
    #         max_seq_len=train_args["max_seq_len"],
    #         dim=train_args["dim"],
    #         depth=train_args["layers"],
    #         heads=train_args["heads"],
    #         causal=True,
    #         nb_features=256,
    #         feature_redraw_interval=1000,
    #         generalized_attention=False,
    #         kernel_fn=torch.nn.ReLU(),
    #         reversible=True,
    #         ff_chunks=10,
    #         use_scalenorm=False,
    #         use_rezero=False,
    #         ff_glu=True,
    #         emb_dropout=train_args["dropout"],
    #         ff_dropout=train_args["dropout"],
    #         attn_dropout=train_args["dropout"],
    #         local_attn_heads=4,
    #         local_window_size=256,
    #         rotary_position_emb=train_args["rel_pos_emb"],
    #         shift_tokens=True,
    #     )
    #     model = performer_pytorch.AutoregressiveWrapper(model)
    # elif train_args["model"] == "linear":
    # default is linear attention
    train_args["model"] = "linear"
    model = linear_attention_transformer.LinearAttentionTransformerLM(
        num_tokens=len(indexer),
        dim=train_args["dim"],
        heads=train_args["heads"],
        depth=train_args["layers"],
        max_seq_len=train_args["max_seq_len"],
        causal=True,
        ff_dropout=train_args["dropout"],
        attn_dropout=train_args["dropout"],
        attn_layer_dropout=train_args["dropout"],
        blindspot_size=64,
        n_local_attn_heads=train_args["heads"],
        local_attn_window_size=128,
        reversible=True,
        ff_chunks=2,
        ff_glu=True,
        attend_axially=False,
        shift_tokens=True,
        use_rotary_emb=train_args["rel_pos_emb"],
    )
    model = linear_attention_transformer.AutoregressiveWrapper(model)
    model = model.to(device)

    # Load the checkpoint
    checkpoint_dir = args.out_dir / "checkpoints"
    if args.model_steps is None:
        checkpoint_filename = checkpoint_dir / "best_model.pt"
    else:
        checkpoint_filename = checkpoint_dir / f"model_{args.model_steps}.pt"
    model.load_state_dict(torch.load(checkpoint_filename, map_location=device))
    logging.info(f"Loaded the model weights from: {checkpoint_filename}")
    model.eval()

    # Get special tokens
    sos = indexer["start-of-song"]
    # son = indexer["start-of-notes"]
    eos = indexer["end-of-song"]
    print(f"eos = {eos}")
    # Get the logits filter function
    if args.filter == "top_k":
        filter_logits_fn = x_transformers.autoregressive_wrapper.top_k
    elif args.filter == "top_p":
        filter_logits_fn = x_transformers.autoregressive_wrapper.top_p
    elif args.filter == "top_a":
        filter_logits_fn = x_transformers.autoregressive_wrapper.top_a
    else:
        raise ValueError("Unknown logits filter.")

    # Iterate over the dataset
    # print(f"len of test_laoder = {len(test_loader)}")
    # with torch.no_grad():
    #     data_iter = iter(test_loader)
    #     for i in tqdm.tqdm(range(args.n_samples), ncols=80):
    #         batch = next(data_iter)
    #         print(f"current batch is = {batch}")
    # batch = next(data_iter)
    results = defaultdict(list)

    with torch.no_grad():
        data_iter = iter(test_loader)
    #     # try:
    # while True:
        for i in tqdm.tqdm(range(args.n_samples), ncols=80):
            batch = next(data_iter)
            # text_emb =  batch["text"].to(device)
            # print(f"shpae = {text_emb.shape}") # 1, 384
            # text_emb = np.load(f"/data2/weihanx/musicgpt/sentence_emb/{sentence_name}.npy")
            # text_emb = np.load("/data2/weihanx/musicgpt/orchestra.npy")
            # text_emb = np.load("/data2/weihanx/musicgpt/videogame.npy")
            # change to three cases: cozy, mjpiano, sadmood
            # text_emb = torch.from_numpy(text_emb).to(device).unsqueeze(0)
            # print(f"shpae = {text_emb.shape}") # 1, 384
            # print(f"number = {i}, current batch is = {batch['name']} ")
            # ------------
            # Ground truth
            # ------------
            truth_np = batch["seq"][0].numpy()
            save_result(
                f"{i}_truth",
                truth_np,
                sample_dir,
                encoding,
                vocabulary,
                representation,
            )
            truth_np = batch["seq"][0].numpy()
            # print(f"shape of truth = {truth_np}")
            result = evaluate(
                truth_np,
                encoding,
                vocabulary,
                representation,
                f"{i}_0",
                eval_dir / "truth",
            )
            results["truth"].append(result)

            # ------------------------
            # Unconditioned generation
            # ------------------------

    #         if train_args["model"] == "compact": # for mmt
    #             # Get output start tokens
    #             tgt_start = torch.zeros(
    #                 (1, 1, 7), dtype=torch.long, device=device
    #             )
    #             tgt_start[:, 0, 0] = sos

    #             # Generate new samples
    #             generated = model.generate(
    #                 tgt_start,
    #                 text_emb,
    #                 args.seq_len,
    #                 # eos_token=eos,
    #                 temperature=args.temperature,
    #                 filter_logits_fn=args.filter,
    #                 filter_thres=args.filter_threshold,
    #                 monotonicity_dim=("type", "beat"),
    #             )
    #         else:

    #             # text should be passed into the model, it is not a prefix
               
    #             tgt_start = torch.zeros(
    #                 (1, 1), dtype=torch.long, device=device
    #             ) # 1*1
    #             # can change number of sample
    #             tgt_start[:, 0] = sos  # batch_size, sequence  # indexr[start-of-song]
    #             generated = model.generate(
    #                 tgt_start,
    #                 text_emb,
    #                 args.seq_len,
    #                 eos_token=eos,
    #                 temperature=args.temperature,
    #                 filter_logits_fn=filter_logits_fn,
    #                 filter_thres=args.filter_threshold,
    #             )
    #         generated_np = torch.cat((tgt_start, generated), 1).cpu().numpy()
    #         eval_reuslt = evaluate(
    #             generated_np[0],
    #             encoding,
    #             vocabulary,
    #             representation,
    #             f"{i}_0",
    #             eval_dir / "conditioned",
    #         )
    #         results["conditioned"].append(eval_reuslt)
    #         # print(f"shape of generated np = {generated_np.shape}") # one row 
    #         # Save the results probably due to "unconditioned"
    #         save_result(
    #             f"{i}_conditioned",
    #             generated_np[0],
    #             sample_dir,
    #             encoding,
    #             vocabulary,
    #             representation,
    #         )
    # for exp, result in results.items():
    #     logging.info(exp)
    #     for key in result[0]:
    #         logging.info(
    #             f"{key}: mean={np.nanmean([r[key] for r in result]):.4f}, "
    #             f"stddev={np.nanstd([r[key]for r in result]):.4f}"
    #         )
    #     # except StopIteration:
    #     #     pass


if __name__ == "__main__":
    main()
