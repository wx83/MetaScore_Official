# generate wav from mid
import muspy
import pathlib 
from pathlib import Path
import random
# this need the environment for muspy

# also calcalte midi result


# evaluate text2midi    


import muspy
from pathlib import Path

def generate_wav_from_mid(audio_data_dir, audio_out_dir, text_data_dir, prompt_list_name):
    # Map problematic key signatures to Mido-compatible names
    key_fix_map = {
        10: 9,  # A# major -> Bb major
        3: 2,   # D# major -> Eb major
        8: 7,   # G# major -> Ab major
        1: 0,   # C# major -> Db major
        6: 5,   # F# major -> Gb major
    }
    for prompt in prompt_list_name:
        print(f"Processing prompt: {prompt}")

        for number in range(0, 10):  # Generate 9 samples
            midi_path = Path(audio_data_dir) / prompt / f"sample_{number}.mid"
            wav_num = number + 1
            audio_data_path = Path(audio_out_dir) / prompt / f"sample{wav_num}.wav"
            audio_data_path.parent.mkdir(parents=True, exist_ok=True)

            if not midi_path.exists():
                print(f"Warning: MIDI file {midi_path} not found. Skipping.")
                continue

            # Read the MIDI file
            music = muspy.read_midi(midi_path)
            # remove key signature, they do not affect playback
            music.key_signatures = []

            muspy.write_audio(audio_data_path, music)
            print(f"Successfully generated: {audio_data_path}")


if __name__ == "__main__":
    audio_data_dir = Path("/data2/weihanx/musicgpt/MetaScore_Official/objective_eval/bart")
    text_data_dir = Path("/data2/weihanx/musicgpt/MetaScore_Official/objective_eval/text_prompt")
    prompt_list_name = ["cheerchristmas", "electronic", "energetic", "gg", "mozart", "piano", "rocksong", "sadmood", "sadpop", "softlove"]
    # audio_out_dir = Path("/data2/weihanx/musicgpt/Supplementary/audio") # for subjective evaluation
    audio_out_dir = Path("/data2/weihanx/musicgpt/MetaScore_Official/objective_eval/bart_wav")
    generate_wav_from_mid(audio_data_dir, audio_out_dir, text_data_dir, prompt_list_name)
    
    # avg_pitch_class_entropy, avg_scale_consistency, avg_groove_consistency = midi_eval(audio_data_dir, prompt_list_name)
    # print(f"avg_pitch_class_entropy = {avg_pitch_class_entropy}")
    # print(f"avg_scale_consistency = {avg_scale_consistency}")
    # print(f"avg_groove_consistency = {avg_groove_consistency}")
