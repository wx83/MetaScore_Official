"""Representation utilities."""
import pathlib
import pprint

import muspy
import numpy as np
import logging
import representation_utils
import utils
import pickle
import json
from pathlib import Path
# Configuration
## need to change back to 12
RESOLUTION = 12
MAX_BEAT = 1024
MAX_DURATION = 384
tag_name = None # condition on tag
program_cond =  True # condition on program
# Duration
KNOWN_DURATIONS = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    15,
    16,
    18,
    20,
    21,
    24,
    30,
    36,
    40,
    42,
    48,
    60,
    72,
    84,
    96,
    120,
    144,
    168,
    192,
    384,
]
DURATION_MAP = {
    i: KNOWN_DURATIONS[np.argmin(np.abs(np.array(KNOWN_DURATIONS) - i))]
    for i in range(1, MAX_DURATION + 1)
}

# Instrument
PROGRAM_INSTRUMENT_MAP = {
    # Pianos
    0: "piano",
    1: "piano",
    2: "piano",
    3: "piano",
    4: "electric-piano",
    5: "electric-piano",
    6: "harpsichord",
    7: "clavinet",
    # Chromatic Percussion
    8: "celesta",
    9: "glockenspiel",
    10: "music-box",
    11: "vibraphone",
    12: "marimba",
    13: "xylophone",
    14: "tubular-bells",
    15: "dulcimer",
    # Organs
    16: "organ",
    17: "organ",
    18: "organ",
    19: "church-organ",
    20: "organ",
    21: "accordion",
    22: "harmonica",
    23: "bandoneon",
    # Guitars
    24: "nylon-string-guitar",
    25: "steel-string-guitar",
    26: "electric-guitar",
    27: "electric-guitar",
    28: "electric-guitar",
    29: "electric-guitar",
    30: "electric-guitar",
    31: "electric-guitar",
    # Basses
    32: "bass",
    33: "electric-bass",
    34: "electric-bass",
    35: "electric-bass",
    36: "slap-bass",
    37: "slap-bass",
    38: "synth-bass",
    39: "synth-bass",
    # Strings
    40: "violin",
    41: "viola",
    42: "cello",
    43: "contrabass",
    44: "strings",
    45: "strings",
    46: "harp",
    47: "timpani",
    # Ensemble
    48: "strings",
    49: "strings",
    50: "synth-strings",
    51: "synth-strings",
    52: "voices",
    53: "voices",
    54: "voices",
    55: "orchestra-hit",
    # Brass
    56: "trumpet",
    57: "trombone",
    58: "tuba",
    59: "trumpet",
    60: "horn",
    61: "brasses",
    62: "synth-brasses",
    63: "synth-brasses",
    # Reed
    64: "soprano-saxophone",
    65: "alto-saxophone",
    66: "tenor-saxophone",
    67: "baritone-saxophone",
    68: "oboe",
    69: "english-horn",
    70: "bassoon",
    71: "clarinet",
    # Pipe
    72: "piccolo",
    73: "flute",
    74: "recorder",
    75: "pan-flute",
    76: None,
    77: None,
    78: None,
    79: "ocarina",
    # Synth Lead
    80: "lead",
    81: "lead",
    82: "lead",
    83: "lead",
    84: "lead",
    85: "lead",
    86: "lead",
    87: "lead",
    # Synth Pad
    88: "pad",
    89: "pad",
    90: "pad",
    91: "pad",
    92: "pad",
    93: "pad",
    94: "pad",
    95: "pad",
    # Synth Effects
    96: None,
    97: None,
    98: None,
    99: None,
    100: None,
    101: None,
    102: None,
    103: None,
    # Ethnic
    104: "sitar",
    105: "banjo",
    106: "shamisen",
    107: "koto",
    108: "kalimba",
    109: "bag-pipe",
    110: "violin",
    111: "shehnai",
    # Percussive
    112: None,
    113: None,
    114: None,
    115: None,
    116: None,
    117: "melodic-tom",
    118: "synth-drums",
    119: "synth-drums",
    120: None,
    # Sound effects
    121: None,
    122: None,
    123: None,
    124: None,
    125: None,
    126: None,
    127: None,
    128: "drums",
}
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

#####################
### use label map ### 
#####################
GENRE_MAP = {
    'Classical/Traditional':0,  # Classical/Traditional
    'Soundtrack/Stage':1,       # Soundtrack/Stage
    'Folk/Country':2,           # Folk/Country
    'Rock/Metal':3,             # Rock/Metal
    'Urban':4,                  # Urban
    'Electronic/Dance':5,       # Electronic/Dance
    'World':6,                  # World
    'Jazz/Blues':7              # Jazz/Blues
}

COMPOSER_MAP = {
    'wolfgang amadeus mozart': 0,
    'william marshall': 1,
    'ludwig van beethoven': 2,
    'ronald karle': 3,
    'claudio miranda': 4,
    'franz schubert': 5,
    'johannes brahms': 6,
    'alexander walker': 7,
    'michael jackson': 8,
    'billie eilish': 9,
    'marc sabatella': 10,
    'the beatles': 11,
    'turlough o\'carolan': 12,
    'franz liszt': 13,
    'tielman susato': 14,
    'robert schumann': 15,
    'william james kirkpatrick': 16,
    'claude debussy': 17,
    'john robson sweney': 18,
    'george gershwin': 19,
    'kristian kalmanlehto': 20,
    'gabriel fauré': 21,
    'paul brouwer': 22,
    'william howard doane': 23,
    'bruce shawyer': 24,
    'joseph haydn': 25,
    'evan dubose': 26,
    'maurice ravel': 27,
    'chopin': 28,
    'adele': 29,
    'richard wagner': 30,
    'allen maples': 31,
    'faith ramsey': 32,
    'david manzanares viles': 33,
    'luca marenzio': 34,
    'franz gruber': 35,
    'felix mendelssohn': 36,
    'javier iriso': 37,
    'ira david sankey': 38,
    'erik satie': 39,
    'ilario pia': 40,
    'rudolf de grijs': 41,
    'charles adams': 42,
    'lady gaga': 43,
    'robert petrie': 44,
    'js bach': 45,
    'glimmer phosphorescence': 46,
}


KNOWN_PROGRAMS = list(
    k for k, v in INSTRUMENT_PROGRAM_MAP.items() if v is not None
)
KNOWN_INSTRUMENTS = list(dict.fromkeys(INSTRUMENT_PROGRAM_MAP.keys()))

KNOWN_EVENTS = ["start-of-song", "end-of-song"]
# if label_name!= None and label_name!="program":
#     KNOWN_EVENTS.extend(f"creators_label_{label}" for label in KNOWN_LABLES)
KNOWN_EVENTS.extend(f"beat_{i}" for i in range(MAX_BEAT))
KNOWN_EVENTS.extend(f"position_{i}" for i in range(RESOLUTION))
KNOWN_EVENTS.extend(
    f"instrument_{instrument}" for instrument in KNOWN_INSTRUMENTS
)
KNOWN_EVENTS.extend(f"drum_{i}" for i in range(35, 82))
KNOWN_EVENTS.extend(f"pitch_{i}" for i in range(128))
KNOWN_EVENTS.extend(f"duration_{i}" for i in KNOWN_DURATIONS)
KNOWN_EVENTS.extend(["start-of-tags"])  # start of tags
# Genre Label
KNOWN_EVENTS.extend(["start-of-genre"]) 
KNOWN_LABLES = list(dict.fromkeys(GENRE_MAP.keys()))
KNOWN_EVENTS.extend(f"tag_genre_{tag}" for tag in KNOWN_LABLES)
KNOWN_EVENTS.extend(["tag_genre_None"])
# Composer Label
KNOWN_LABLES = list(dict.fromkeys(COMPOSER_MAP.keys()))
KNOWN_EVENTS.extend(["start-of-composer"]) 
KNOWN_EVENTS.extend(f"tag_composer_{composer}" for composer in KNOWN_LABLES) # using instrument program map
KNOWN_EVENTS.extend(["tag_composer_None"])

# Instrument Label
KNOWN_EVENTS.extend(["start-of-instrument"]) 
KNOWN_EVENTS.extend(f"tag_instrument_{instrument}" for instrument in KNOWN_INSTRUMENTS) # using instrument program map
KNOWN_EVENTS.extend(["tag_instrument_None"])

# Complexity Label
KNOWN_EVENTS.extend(["start-of-complexity"]) 
KNOWN_EVENTS.extend(f"tag_complexity_{i}" for i in range(0,6)) # using instrument program map
KNOWN_EVENTS.extend(["tag_complexity_None"]) # using instrument program map

KNOWN_EVENTS.extend(["start-of-notes"]) # add at the end

EVENT_CODE_MAPS = {event: i for i, event in enumerate(KNOWN_EVENTS)}
CODE_EVENT_MAPS = utils.inverse_dict(EVENT_CODE_MAPS)

# print(f"EVENT_CODE_MAPS= {EVENT_CODE_MAPS}")

class Indexer:
    def __init__(self, data=None, is_training=False):
        self._dict = dict() if data is None else data
        self._is_training = is_training

    def __getitem__(self, key):
        if self._is_training and key not in self._dict:
            self._dict[key] = len(self._dict)
            return len(self._dict) - 1
        return self._dict[key]

    def __len__(self):
        return len(self._dict)

    def __contain__(self, item):
        return item in self._dict

    def get_dict(self):
        """Return the internal dictionary."""
        return self._dict

    def train(self):
        """Set training mode."""
        self._is_training = True

    def eval(self):
        """Set evaluation mode."""
        self._is_learning = False


def get_encoding():
    """Return the encoding configurations."""
    return {
        "resolution": RESOLUTION,
        "max_beat": MAX_BEAT,
        "max_duration": MAX_DURATION,
        "program_instrument_map": PROGRAM_INSTRUMENT_MAP,
        "instrument_program_map": INSTRUMENT_PROGRAM_MAP,
        "genre_map": GENRE_MAP,
        "composer_map":COMPOSER_MAP,
        "duration_map": DURATION_MAP,
        "event_code_map": EVENT_CODE_MAPS,
        "code_event_map": CODE_EVENT_MAPS,
    }


def load_encoding(filename):
    """Load encoding configurations from a JSON file."""
    encoding = utils.load_json(filename)
    for key in ("program_instrument_map", "code_event_map", "duration_map"):
        encoding[key] = {
            int(k) if k != "null" else None: v
            for k, v in encoding[key].items()
        }
    return encoding


def encode_notes(notes, encoding, indexer, tag_dict_path=None):
    """Encode the notes into a sequence of code tuples.

    Each row of the output is encoded as follows.

        (event_type, beat, position, pitch, duration, instrument)

    """
    # tag_dict is a json path
    # try:
        # with open(tag_dict_path, "r") as file:
        #     tag_dict = json.load(file)


        # assert tag_dict is not None
        # # Get maps
        # duration_map = encoding["duration_map"]
        # program_instrument_map = encoding["program_instrument_map"]
        # genre_map = encoding["genre_map"]
        # composer_map = encoding["composer_map"]
        # # The seq input should be <sos> programs <son> notes <eos>
        # # Start the codes with an SOS event

        # codes.append(indexer["start-of-tags"])
        # # tag_dict = {"genre":[], "composer":[], "complexity":[], "favoriate": []}
        # for key, value in tag_dict.items(): # loop through every pair
        #     if key == "genre":
        #         codes.append(indexer['start-of-genre'])
        #         for val in value:
        #             if val is not None: 
        #                 codes.append(indexer[f"tag_genre_{val}"])
        #                 # print(f"dbug")
        #                 # print(indexer[f"tag_genre_{val}"])
        #             else:# if the value if none
        #                 codes.append(indexer["tag_genre_None"])
        #     elif key == "composer":
        #         codes.append(indexer['start-of-composer'])
        #         for val in value:
        #             if val is not None: 
        #                 codes.append(indexer[f"tag_composer_{val}"])
        #                 # logging.info(f"find composer = {val}")
        #             else:# if the value if none

        #                 codes.append(indexer["tag_composer_None"])
        
        #     elif key == "complexity":
        #         codes.append(indexer['start-of-complexity'])
        #         for val in value:
        #             if val is not None: 
        #                 codes.append(indexer[f"tag_complexity_{val}"])
        #             else:
        #                 codes.append(indexer['tag_complexity_None']) 
        # codes.append(indexer['start-of-instrument'])
        # program_list  = []
        # for note in notes:
        #     if note[4] <= 128: # keep drum track
        #         instrument = program_instrument_map[note[4]]
        #         # print(f"instrument = {instrument}")
        #         if instrument is None:
        #             continue
        #         program_list.append(instrument)
        # program_list_set = set(program_list)
        #     # print(f"tag = {tag_dict_path}, program = {program_list_set}")
        # for c in program_list_set:
        #     codes.append(indexer[f"tag_instrument_{c}"])     


        # # print(f"codes after trans = {codes}") # check has instrument and complexity
        # # complexity


        # codes.append(indexer["start-of-notes"])
        # Encode the notes
    #10/4 to generate GT
    duration_map = encoding["duration_map"]
    program_instrument_map = encoding["program_instrument_map"]
    genre_map = encoding["genre_map"]
    composer_map = encoding["composer_map"]
    max_beat = encoding["max_beat"]
    max_duration = encoding["max_duration"]
    codes = [indexer["start-of-song"]]
    last_beat = 0
    for beat, position, pitch, duration, program in notes:
        # Skip if max_beat has reached
        if beat > max_beat:
            continue
        # skip some wired case
        if pitch > 127 or pitch < 0:
            logging.debug(f"error tag_dict_path = {tag_dict_path}")
            print(f"file name remove = {tag_dict_path}")
            continue
        # Skip unknown instruments
        if program < 128:
            instrument = program_instrument_map[program]
            # print(f"original instrument = {instrument}")
            if instrument is None:
                continue
        else:
            if pitch < 35 or pitch > 81:
                continue
            instrument = "drums"
        if beat > last_beat:
            codes.append(indexer[f"beat_{beat}"])
            last_beat = beat
        codes.append(indexer[f"position_{position}"])
        codes.append(indexer[f"instrument_{instrument}"])
        # print(f"instrument append") # matched
        # print(indexer[f"instrument_{instrument}"])
        if instrument == "drums":
            codes.append(indexer[f"drum_{pitch}"])
        else:
            if pitch > 127 or pitch < 0:
                print(f"error tag_dict_path = {tag_dict_path}")
            codes.append(indexer[f"pitch_{pitch}"])
            duration = duration_map[min(duration, max_duration)]
            codes.append(indexer[f"duration_{duration}"])
    # print(f"c = {c}")
    # End the codes with an EOS event
    codes.append(indexer["end-of-song"])
    # print(f"sequence code = {}")

    # print(f"codes = {len(codes)}")
    return np.array(codes)
    # except:
    #     print(f"tag_dict_pth = {tag_dict_path}")


def encode(music, encoding, indexer,tag_dict):
    """Encode a MusPy music object into a sequence of codes.

    Each row of the input is encoded as follows.

        (event_type, beat, position, pitch, duration, instrument)

    """
    # Extract notes
    notes = representation_utils.extract_notes(music, encoding["resolution"])
    ### EXTRACT TAGS
    codes = encode_notes(notes, encoding, indexer, tag_dict)

    return codes


def decode_notes(data, encoding, vocabulary):
    # the notes begins after son index
    """Decode codes into a note sequence."""
    # Get variables and maps
    duration_map = encoding["duration_map"]
    program_instrument_map = encoding["program_instrument_map"]
    instrument_program_map = encoding["instrument_program_map"]
    genre_map = encoding["genre_map"]
    composer_map = encoding["composer_map"]

    prog = []
    label = []
    # Initialize variables
    beat = 0
    position = None
    program = None
    pitch = None
    duration = None

    # Decode the codes into a sequence of notes
    notes = []
    for code in data:
        event = vocabulary[code]
        # the notes start with "start-of-notes"
        
        if event == "start-of-song":
            continue
        elif event.startswith("program_label"): # The condition cannot be translated into music
            continue
        elif event == "start-of-program":
            # program = int(event.split("_")[1])
            # prog.append(program)
            continue
        elif event == "start-of-tags":
            continue
        elif event.startswith("tag"): # tag_genre_, tag_composer_
            continue
        elif event == "start-of-genre":
            continue
        elif event == "start-of-composer":
            continue
        elif event == "start-of-instrument":
            continue
        elif event == "start-of-complexity":
            continue
        elif event == "start-of-notes":
            continue
        elif event == "end-of-song":
            break
        elif event.startswith("beat"):
            beat = int(event.split("_")[1]) # beat line
            # Reset variables
            position = None
            program = None
            pitch = None
            duration = None
        elif event.startswith("position"):
            position = int(event.split("_")[1])
            # Reset variables
            program = None
            pitch = None
            duration = None
        elif event.startswith("instrument"):
            instrument = event.split("_")[1]
            if instrument == "drums":
                program = 128 
            else:
                program = instrument_program_map[instrument]
        elif event.startswith("pitch"):
            pitch = int(event.split("_")[1])
        elif event.startswith("duration"):
            duration = int(event.split("_")[1])
            if (
                position is None
                or program is None
                or pitch is None
                or duration is None
            ):
                continue
            notes.append((beat, position, pitch, duration, program))
        elif event.startswith("drum"): # drum_pitch
            if position is None or program != 128:
                continue
            # print(f"found drum = {notes[-1]}")
            pitch = int(event.split("_")[1])
            notes.append((beat, position, pitch, 1, 128)) # if it is a drum track, append
        else:
            raise ValueError(f"Unknown event type for: {event}")

    return  notes


def decode(codes, encoding, vocabulary):
    """Decode codes into a MusPy Music object.

    Each row of the input is encoded as follows.

        (event_type, beat, position, pitch, duration, instrument)

    """
    # Get resolution
    resolution = encoding["resolution"]

    # Decode codes into a note sequence
    notes = decode_notes(codes, encoding, vocabulary)

    # Reconstruct the music object
    music = representation_utils.reconstruct(notes, resolution)

    return music


def dump(data, vocabulary):
    """Decode the codes and dump as a string."""
    # Iterate over the rows
    lines = []
    # print(f"data = {data.shape}") # (356, ), array
    for code in data:
        # print(f"dump code = {code}")
        event = vocabulary[code]
        # print(f"my event is = {event}")
        if(event == "start-of-song"
            ):
            continue
        elif(event == "start-of-tags"):
            lines.append(event)
        elif (event == "start-of-genre"):
            lines.append(event)
        elif (event.startswith("tag")): # all four tag: tag_instrument, tag_genre, tag_composer, tag_complexity
            lines.append(event)
        elif (event == "start-of-composer"):
            lines.append(event)
        elif (event == "start-of-instrument"):
            lines.append(event)          
        elif (event == "start-of-complexity"):
            lines.append(event)          
        elif (
            event == "start-of-notes"
            or event.startswith("beat")
            or event.startswith("position")
        ):
            lines.append(event)
        elif event == "end-of-song":
            lines.append(event)
            break
        elif (
            event.startswith("instrument")
            or event.startswith("pitch")
            or event.startswith("duration")
            or event.startswith("drum")
        ):
            lines[-1] = f"{lines[-1]} {event}"
        else:
            raise ValueError(f"Unknown event type for: {event}")
        # print(f"Lines = {lines} \n")
    return "\n".join(lines)


def save_txt(filename, data, vocabulary):
    """Dump the codes into a TXT file."""
    with open(filename, "w") as f:
        f.write(dump(data, vocabulary))


def save_csv_codes(filename, data):
    """Save the representation as a CSV file."""
    assert data.ndim == 1
    np.savetxt(
        filename,
        data,
        fmt="%d",
        delimiter=",",
        header="code",
        comments="",
    )


def main():
    """Main function."""
    # Get the encoding
    encoding = get_encoding()
    encoding_file_name = "encoding_fourtag.json"
    filename = pathlib.Path(__file__).parent / encoding_file_name
    utils.save_json(filename, encoding)

    # Load encoding
    encoding = load_encoding(filename)

    # Print the maps
    print(f"{' Maps ':=^40}")
    for key, value in encoding.items():
        if key in ("program_instrument_map", "instrument_program_map"):
            print("-" * 40)
            print(f"{key}:")
            pprint.pprint(value, indent=2)

    # Print the variables
    print(f"{' Variables ':=^40}")
    print(f"resolution: {encoding['resolution']}")
    print(f"max_beat: {encoding['max_beat']}")
    print(f"max_duration: {encoding['max_duration']}")

    # Load the example
    music = muspy.load(pathlib.Path(__file__).parent / "example.json")
    # # print(f"label path = {pathlib.Path(__file__).parent}")
    # if tag_name != None:
    #     f = open('/data2/weihanx/musicgpt/example.json')
    #     label_json = json.load(f)
    # if tag_name == None:
    #     label_json = None
    # # for i in label_json['metadata']:
    # #     print(f"metadata = {i}")
    # # print(f"matadata composer = {label_json['metadata']['creators']}")
    # # label_json = json.load("/data2/weihanx/musicgpt/example.json")
    # # print(f"loaded muisc = {label_json}")
    # # Get the indexer
    indexer = Indexer(encoding["event_code_map"]) # event code map is the final indxer
    tag_dict_dir = Path("musescore_tag_comb/a/6")
    tag_dict = tag_dict_dir/"Qma6eBiZYiyUQ6jCghXUSCaN9NxkLHw52FhRPXq7obzb3F.json"
    
    # print(f"tag_dict = {tag_dict}")
    

    # Encode the music
    encoded = encode(music, encoding, indexer, tag_dict)
    print(f"Codes:\n{encoded}")

    # # Get the learned vocabulary
    # vocabulary = utils.inverse_dict(indexer.get_dict())

    # print("-" * 40)
    # print(f"Decoded:\n{dump(encoded, vocabulary)}")

    # music = decode(encoded, encoding, vocabulary)
    # print(f"Decoded musics:\n{music}")


if __name__ == "__main__":
    main()
