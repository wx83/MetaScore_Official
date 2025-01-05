import torch
from transformers import AutoTokenizer, BloomForCausalLM
import json
import pathlib
from pathlib import Path
from utils import load_json, save_json, load_txt, save_txt
import json
import time
import logging
import argparse
import os
from huggingface_hub import HfFolder, InferenceClient, login

inference_client = InferenceClient("bigscience/bloom")
device = torch.device("cuda:1")
# model = model.to(device)
"""
Use template instead
"""
def parse_args():
    parser = argparse.ArgumentParser(description="Tag to text")
    parser.add_argument("--process", type=str, default="logging/tagtotext.txt", help="Logging file path")
    parser.add_argument("--output", type=str, default="logging/tagtotext.txt", help="Logging file path")
    return parser.parse_args()

logging.basicConfig(
    filename="logging/tagtotext.txt",  # Use the logging file path from command-line argument
    filemode='a',  # Append mode
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Log message format
)
logger = logging.getLogger('MyLogger')

def get_bloom_response(
    inference_api,
    prompt,
    max_length=32,
    greedy=False,
    temperature=0.7,
    top_p=0.5,
    no_repeat_ngram_size=2,
    seed=1024,
    stop=["<\\s>"],
):
    """Get the bloom response with a prompt."""
    # Get the Bloom response
    params = {
        "use_gpu": True,
        "wait_for_model": True,
        "use_cache": False,
        "return_full_text": False,
        "max_new_tokens": max_length,
        "seed": seed,
        "do_sample": not greedy,
        "temperature": temperature,
        "top_p": top_p,
        "no_repeat_ngram_size": no_repeat_ngram_size,
    }
    response = inference_client.post(json={"inputs": prompt, "parameters": params}, model = "bigscience/bloom")

    response_decoded = response.decode('utf-8')
    response_json = json.loads(response_decoded) 
    # print(f"response json = {response_json}")
    generated_text = response_json[0]['generated_text']
    # print(f"response = {response_json}")

    # Return None and log errors if encountered
    if "error" in response_json:
        logging.error(response_json["error"])
        return None

    # Return the generated text
    time.sleep(2)
    return generated_text

def switch_genre(genre_type):
    input_string = genre_type
    output_string = input_string.replace("/", " or ")
    return output_string

def get_caption(composer, complexity, instrument, genre, timesig, keysig, tempo, noisy):
    complexity_map = {
        '1': 'easy',
        '2': 'intermediate',
        '3': 'advanced'
    }

    # Ensure all lists are initialized
    composer = composer or []
    complexity = complexity or []
    instrument = instrument or []
    genre = genre or []
    timesig = timesig or []
    keysig = keysig or []
    noisy = noisy or []

    if complexity and complexity[0] is not None:
        complexity = [complexity_map[i] for i in complexity]
    if genre and genre[0] is not None:
        genre = [content.lower() for content in genre]
        genre = [switch_genre(content) for content in genre]
    if composer:
        composer = [comp.title() for comp in composer if comp is not None]
    if timesig and timesig[0] is not None:
        timesig = [f"{timesig[0]} time"]
    if keysig:
        keysig = [keysig[0]]
    if tempo:
        tempo = [f"{tempo[0]} bpm"]

    # print(f"composer = {composer}, complexity = {complexity}, instrument = {instrument}, genre = {genre}, timesig = {timesig}, keysig = {keysig}, noisy = {noisy}")

    if noisy:
        input_seq = composer + complexity + instrument + genre + timesig + keysig + tempo + noisy
    else:
        input_seq = composer + complexity + instrument + genre + timesig + keysig + tempo

    filtered_seq = [item for item in input_seq if item not in [None, '', [], {}] and len(item) < 50]

    input_seq_str = ';'.join(filtered_seq)

    if not input_seq_str:  # 0 btype read
        return None
    # Example cases
    case1 = "Input: Chopin, piano, easy, 4/4 time, C major, 110 bpm. Output: An easy piano piece in Chopin's style, in 4/4 time, C major, with a tempo of 110 bpm. <\\s>"
    case2 = "Input: folk or country, creepy, 3/4 time, E minor, Programmatic, robot. Output: A 3/4 folk or country piece in E minor with a creepy vibe. <\\s>"
    case3 = "Input: classical or traditional, Heaven's, His, Jesus, Lord, Son, accords, dawn, day, grace, hymn, light, peace, rest, soul, sovereign. Output: A classical or traditional hymn. <\\s>"
    case4 = "Input: Michael Jackson, bass, guitar, rock or metal, technical, electronic or dance. Output: A mix of pop, rock, and electronic music featuring Michael Jackson's style with bass and guitar. <\\s>"
    case5 = "Input: William Marshall, Adele, advanced, piano, violin, soundtrack, video game. Output: An advanced piano and violin piece in the style of William Marshall and Adele, suitable for a video game soundtrack. <\\s>"
    case6 = "Input: orchestra, into the unknown, frozen, cover, Rm, 20. Output: An orchestral cover of 'Into the Unknown' by Frozen. <\\s>"

    inference = f"Input: {input_seq_str}. Output: "
    prefix = case1 + case2 + case3 + case4 + case5 + case6 + inference
    prefix_len = len(prefix)
    max_length = 32
    max_length_lim = prefix_len + max_length  # 30 words caption max

    output = get_bloom_response(inference_client, prefix)  # default is 32 max
    output_string = output.split('<\\s>')[0]
    # print(f"output string = {output_string}")
    return output_string



def sample_test(music):
    with open("/data2/weihanx/musicgpt/metascore_full_plus/metascore_full_plus_tag/a/1/Qma1auL7FuyWxJoo6x3hAvD6FuucnsJchxSCwDKzDDF4Tz.json", "r") as file:
        data = json.load(file)
    genre = data['genre']
    composer = data["composer"]
    complexity = data['complexity']
    timesig = data['timesig']
    instrument = data['instrument']
    keysig = data['keysig']
    tempo = data['tempo']

    noisy = load_txt(Path("metadata_noisy_tag") / music[2] / music[3] / f"{music}.txt")
    text = get_caption(composer, complexity, instrument, genre, timesig, keysig, tempo, noisy)
    print(f"text = {text}")

if __name__ == '__main__':
    # sample_test("Qma1auL7FuyWxJoo6x3hAvD6FuucnsJchxSCwDKzDDF4Tz")
    input_dir = Path("/data2/weihanx/musicgpt/metascore_full_plus/metascore_full_plus_tag")
    out_dir = Path("metascore_cap_large_0811")
    out_dir.mkdir(parents=True, exist_ok=True)
    args = parse_args()
    music_name_list = load_txt(args.process)
    valid_music = []
    # sample_test("Qma1MFbxgmWnB2iWZtBikd1sAN2AFdk9kqqLNUGTfJfyae")
    for music in music_name_list:
        try:
            out_path = out_dir / music[2] / music[3] / f"{music}.txt"
            if out_path.exists():
                continue
            with open(input_dir / music[2] / music[3] / f"{music}.json", "r") as file:
                data = json.load(file)
            genre = data['genre']
            composer = data["composer"]
            complexity = data['complexity']
            timesig = data['timesig']
            instrument = data['instrument']
            keysig = data['keysig']
            tempo = data['tempo']
            noisy_path = Path("metadata_noisy_tag") / music[2] / music[3] / f"{music}.txt"
            if noisy_path.exists():
                noisy = load_txt(noisy_path)
            else:
                noisy = None
            text = get_caption(composer, complexity, instrument, genre, timesig, keysig, tempo, noisy)

            if text is not None:
                valid_music.append(music)
                out_path = out_dir / music[2] / music[3] / f"{music}.txt"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w") as file:
                    file.write(text)
            with open(args.output, "w") as file:
                for item in valid_music:
                    file.write(f"{item}\n")
        except Exception as e:
            logger.debug(f"Error processing {music}: {e}")
