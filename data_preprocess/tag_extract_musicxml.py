import xml.etree.ElementTree as ET
from collections import defaultdict
import pathlib
from pathlib import Path
import pickle
from utils import load_json, save_json, load_txt, save_txt
import json
import logging

logging.basicConfig(
    filename="/data2/weihanx/musicgpt/tag_extract_logging/tempo_key.txt",  # Use the logging file path from command-line argument
    filemode='a',  # Append mode
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Log message format
)
logger = logging.getLogger('MyLogger')


start_directory = 'musicxml'
start_path = Path(start_directory)
INSTRUMENT_PROGRAM_MAP = {
    # Pianos
    "piano": 0,
    "electric-piano": 4,
    "harpsichord": 6,
    "clavinet": 7,
    # Chromatic Percussion
    "celesta": 8,
    "glockenspiel": 9,
    "music-box": 10,
    "vibraphone": 11,
    "marimba": 12,
    "xylophone": 13,
    "tubular-bells": 14,
    "dulcimer": 15,
    # Organs
    "organ": 16,
    "church-organ": 19,
    "accordion": 21,
    "harmonica": 22,
    "bandoneon": 23,
    # Guitars
    "nylon-string-guitar": 24,
    "steel-string-guitar": 25,
    "electric-guitar": 26,
    # Basses
    "bass": 32,
    "electric-bass": 33,
    "slap-bass": 36,
    "synth-bass": 38,
    # Strings
    "violin": 40,
    "viola": 41,
    "cello": 42,
    "contrabass": 43,
    "harp": 46,
    "timpani": 47,
    # Ensemble
    "strings": 49,
    "synth-strings": 50,
    "voices": 52,
    "orchestra-hit": 55,
    # Brass
    "trumpet": 56,
    "trombone": 57,
    "tuba": 58,
    "horn": 60,
    "brasses": 61,
    "synth-brasses": 62,
    # Reed
    "soprano-saxophone": 64,
    "alto-saxophone": 65,
    "tenor-saxophone": 66,
    "baritone-saxophone": 67,
    "oboe": 68,
    "english-horn": 69,
    "bassoon": 70,
    "clarinet": 71,
    # Pipe
    "piccolo": 72,
    "flute": 73,
    "recorder": 74,
    "pan-flute": 75,
    "ocarina": 79,
    # Synth Lead
    "lead": 80,
    # Synth Pad
    "pad": 88,
    # Ethnic
    "sitar": 104,
    "banjo": 105,
    "shamisen": 106,
    "koto": 107,
    "kalimba": 108,
    "bag-pipe": 109,
    "shehnai": 111,
    # Percussive
    "melodic-tom": 117,
    "synth-drums": 118,
    # Drums
    "drums": 128,
}
valid_instrument = INSTRUMENT_PROGRAM_MAP.keys()
def get_composer_from_musicxml(file_path):
    # Load the XML content from the file
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Extract the composer's name
        # We look for the 'metaTag' element with the attribute name='composer' and extract its text
        composer_name = root.find(".//metaTag[@name='composer']")
        if composer_name is not None:
            return composer_name.text
        else:
            return "Composer not found"
    except ET.ParseError as e:
        return f"Error parsing XML: {e}"
    except FileNotFoundError as e:
        return f"File not found: {e}"

import xml.etree.ElementTree as ET

def get_instruments_from_musicxml(file_path):
    # Load the XML content from the file
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Initialize a list to collect instrument names
        instrument_names = []

        # Attempt to extract instrument names based on provided structure
        # Adjust the XPath as necessary based on your actual MusicXML structure
        # This is a guess since the provided XML doesn't have filled instrument names
        for instrument in root.findall(".//Part/Instrument/trackName"):
            if instrument.text:
                instrument_names.append(instrument.text)
        
        # Fallback to check another possible tag for instruments if the above is empty
        if not instrument_names:
            for part_name in root.findall(".//Part/trackName"):
                if part_name.text:
                    instrument_names.append(part_name.text)

        if instrument_names:
            return instrument_names
        else:
            return [None]
    except ET.ParseError as e:
        return [f"Error parsing XML: {e}"]
    except FileNotFoundError as e:
        return [f"File not found: {e}"]


def get_timesig_from_musicxml(input_dir, output_dir, file_name):
    # Load the XML content from the file
    try:
        file_path = input_dir / file_name[2] / file_name[3] / f"{file_name}.musicxml"
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        time_signatures = []
        for timesig in root.findall(".//TimeSig"):
            sigN = timesig.find('sigN').text
            sigD = timesig.find('sigD').text
            time_signatures.append(f"{sigN}/{sigD}")
        # print(f"Time signatures: {time_signatures}")
        if len(time_signatures) > 1:
            logger.info(f"Multiple time signatures found for {file_name}")
        output_file = output_dir / file_name[2] / file_name[3] / f"{file_name}.txt"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            for ts in time_signatures:
                f.write(ts + '\n')

    except:
        logger.error(f"Error processing {file_name}")

def convert_to_key(accidental, mode):
    key_map = {
        '0': ("C major", "A minor"),
        '1': ("G major", "E minor"),
        '2': ("D major", "B minor"),
        '3': ("A major", "F# minor"),
        '4': ("E major", "C# minor"),
        '5': ("B major", "G# minor"),
        '6': ("F# major", "D# minor"),
        '7': ("C# major", "A# minor"),
        '-1': ("F major", "D minor"),
        '-2': ("Bb major", "G minor"),
        '-3': ("Eb major", "C minor"),
        '-4': ("Ab major", "F minor"),
        '-5': ("Db major", "Bb minor"),
        '-6': ("Gb major", "Eb minor"),
        '-7': ("Cb major", "Ab minor")
    }

    if mode is not None:
        if mode == "major":
            return key_map[accidental][0]
        else:

            return key_map[accidental][1]

    return key_map[accidental]

def get_keysig_from_musicxml(input_dir, output_dir, file_name):

    try:
        file_path = input_dir / file_name[2] / file_name[3] / f"{file_name}.musicxml"
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        key_signatures = []
        for keysig in root.findall(".//KeySig"):
            accidental = keysig.find('accidental').text
            mode = keysig.find('mode').text
            key = convert_to_key(accidental, mode)
            key_signatures.append(key)
        # print(f"Time signatures: {time_signatures}")
        if len(key_signatures) > 1:
            logger.info(f"Multiple key signatures found for {file_name}")
        output_file = output_dir / file_name[2] / file_name[3] / f"{file_name}.txt"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            for ts in key_signatures:
                # distinct_keys[ts] = distinct_keys.get(ts, 0) + 1
                f.write(ts + '\n')

    except:
        # if either mode or accidental is not found, skip
        logger.error(f"Error processing {file_name}")
    
    return key_signatures

def get_tempo_from_musicxml(input_dir, output_dir, file_name):

    try:
        file_path = input_dir / file_name[2] / file_name[3] / f"{file_name}.musicxml"
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        tempos = []
        for tempo in root.findall(".//Tempo"):
            qpm = tempo.find('tempo').text
            qpm = round(float(qpm) * 60)
            tempos.append(str(qpm))
        # print(f"Time signatures: {time_signatures}")
        if len(tempos) > 1:
            logger.info(f"Multiple time signatures found for {file_name}")
        output_file = output_dir / file_name[2] / file_name[3] / f"{file_name}.txt"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            for ts in tempos:
                f.write(ts + '\n')

    except:
        # if no tempos is found, skip
        logger.error(f"Error processing {file_name}")
    return tempos


def debug_test():
    input_dir = Path('/data2/weihanx/musicgpt/musicxml')
    output_dir = Path('/data2/weihanx/musicgpt')
    file_name = "Qmd1a3ZJ9MP5YJQm2mZZ2koN4r1n6ZJJXJ5ECdEARquKB7"
    get_keysig_from_musicxml(input_dir, output_dir, file_name) # if not found either accidental or mode, skip
    get_tempo_from_musicxml(input_dir, output_dir, file_name) # if not found tempo, skip
    
    print(res)
if __name__ == '__main__':
    debug_test()
    # music_name_lists = load_txt('/data2/weihanx/musicgpt/metascore_large.txt')
    # input_dir = Path('/data2/weihanx/musicgpt/musicxml')
    # output_dir = Path('/data2/weihanx/musicgpt/metadata_tag_timesig')
    # tempo_output_dir = Path('/data2/weihanx/musicgpt/metadata_tag_tempo')
    # key_output_dir = Path('/data2/weihanx/musicgpt/metadata_tag_key')
    # tempo_output_dir.mkdir(parents=True, exist_ok=True)
    # key_output_dir.mkdir(parents=True, exist_ok=True)
    # distinct_keys = {}
    # distinct_tempo = {}
    # distinct_tempo_init = {}
    # for music in music_name_lists:
    #     print(f"music = {music}")
    #     # get_timesig_from_musicxml(input_dir, output_dir, music)
    #     tempo_list = get_tempo_from_musicxml(input_dir, tempo_output_dir, music)
    #     key_list = distinct_key = get_keysig_from_musicxml(input_dir, key_output_dir, music)
    #     if tempo_list and key_list:
    #         for t in tempo_list:
    #             distinct_tempo[t] = distinct_tempo.get(t, 0) + 1
    #         for k in key_list:
    #             distinct_keys[k] = distinct_keys.get(k, 0) + 1
    #         distinct_tempo_init[tempo_list[0]] = distinct_tempo_init.get(tempo_list[0], 0) + 1
            
    # save_json('/data2/weihanx/musicgpt/metadata_tag_tempo_dist.json', distinct_tempo)
    # save_json('/data2/weihanx/musicgpt/metadata_tag_tempo_init_dist.json', distinct_tempo_init)
    # save_json('/data2/weihanx/musicgpt/metadata_tag_key_dist.json', distinct_keys)
        # break
     
 