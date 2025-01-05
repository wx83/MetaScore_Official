"""
This file is used to extract other tags in the musescore-metadata
"""

# check the correspondence between metadata and musescore song name:

import pickle
from utils import load_txt, save_txt, load_json, save_json
import pathlib
from pathlib import Path
import json
import matplotlib.pyplot as plt
from collections import defaultdict

# Load the pickle object
"""
Example: 8113644471849408.json: QmYp16YSxVyfggQbSrAC149K1B3C26G4ZawYeroZU18Gp5
"""
with open('map_json_mscz.pkl', 'rb') as file:
    song_name_dict = pickle.load(file)


def get_composer():
    input_dir = Path('/data2/weihanx/musicgpt/metadata')
    out_dir = Path('/data2/weihanx/musicgpt/metadata_tag_composer_raw')
    music_name_list = Path("/data2/weihanx/musicgpt/musescore_path.txt")
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
    music_names_list = load_txt(music_name_list)
    failed = []
    for music in music_names_list:
        try:
            music = Path(music)
            metadata_name = music.name # 1622031.json

            song_name = song_name_dict[metadata_name]
            print(f"song_name = {song_name}")
            meta_json = input_dir / f"{music}"

            with open(meta_json, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            count_view_set = set()

            # Check the first potential source
            count_view_1 = json_data['data'].get('score', {}).get('composer_name')
            if count_view_1:
                if isinstance(count_view_1, list):
                    count_view_set.update(count_view_1)
                else:
                    count_view_set.add(count_view_1)

            # Check the second potential source
            count_view_2 = json_data['data'].get('composer')
            if count_view_2:
                if isinstance(count_view_2, list):
                    count_view_set.update(count_view_2)
                else:
                    count_view_set.add(count_view_2)

            save_path = out_dir / song_name[2] / song_name[3] / f"{song_name}.txt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_txt(save_path, list(count_view_set))

        except Exception as e:
            print(f"Error processing {music}: {e}")
            failed.append(music)
            continue
    save_txt("failed_raw_composer.txt", failed)


def get_rating():
    input_dir = Path('/data2/weihanx/musicgpt/metadata')
    out_dir = Path('/data2/weihanx/musicgpt/metadata_rating')
    music_name_list = Path("/data2/weihanx/musicgpt/musescore_path.txt")
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
    music_names_list = load_txt(music_name_list)
    failed = []
    # rating_data = defaultdict(set/)
    low_rate_file = [] # < 3
    high_rate_file = [] # > 4
    mid_rate_file = []  # 3-4
    for music in music_names_list:
        try:
            music = Path(music)
            metadata_name = music.name # 1622031.json

            song_name = song_name_dict[metadata_name]
            print(f"song_name = {song_name}")
            meta_json = input_dir / f"{music}"

            with open(meta_json, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            # count_view_set = set() # it can only have one rate

            # Check the first potential source
            rating = json_data['data'].get('score', {}).get('rating')['rating']
            print(f"rating = {rating}")
            if rating < 3:
                low_rate_file.append(song_name)
            elif rating > 4:
                high_rate_file.append(song_name)
            else:
                mid_rate_file.append(song_name)
            



            save_path = out_dir / song_name[2] / song_name[3] / f"{song_name}.txt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(str(rating))

        except Exception as e:
            print(f"Error processing {music}: {e}")
            failed.append(music)
            continue
    save_txt("low_rate_file.txt", low_rate_file)
    save_txt("mid_rate_file.txt", mid_rate_file)
    save_txt("high_rate_file.txt", high_rate_file)
    save_txt("failed_rating.txt", failed)

def get_original_genre():
    input_dir = Path('/data2/weihanx/musicgpt/metadata')
    out_dir = Path('/data2/weihanx/musicgpt/metadata_tag_genre_raw')
    music_name_list = Path("/data2/weihanx/musicgpt/musescore_path.txt")
    with_genre = Path("/data2/weihanx/musicgpt/process_name/metascore_w_genre.txt")
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
    music_names_list = load_txt(music_name_list)
    failed = []
    for music in music_names_list:
        try:
            music = Path(music)
            metadata_name = music.name # 1622031.json

            song_name = song_name_dict[metadata_name]
            if song_name not in load_txt(with_genre):
                continue
            meta_json = input_dir / f"{music}"

            with open(meta_json, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            count_view_set = set()

            # Check the second potential source
            count_view_2 = json_data['data'].get('genres')
            print(f"count_view_2 = {count_view_2}")
            if count_view_2:
                if isinstance(count_view_2, list):
                    for genre in count_view_2:
                        count_view_set.add(genre['name'])
                else:
                    count_view_set.add(count_view_2)

            save_path = out_dir / song_name[2] / song_name[3] / f"{song_name}.txt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_txt(save_path, list(count_view_set))
        except Exception as e:
            print(f"Error processing {music}: {e}")
            failed.append(music)
        continue
    save_txt("failed_raw_genre.txt", failed)

if __name__ == '__main__':
    # get_original_genre()
    get_rating()
    # get_composer