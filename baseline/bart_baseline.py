import torch
from samplings import top_p_sampling, temperature_sampling
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained('sander-wood/text-to-music')
model = AutoModelForSeq2SeqLM.from_pretrained('sander-wood/text-to-music')
model = model

max_length = 1024
top_p = 0.9
temperature = 1.0

# text = "This music piece is in a sad mood.."
text_container = [
    "A cheerful christmas song suitable for children.",
    "This is a soundtrack with electronic/dance vibe.",
    "An energetic and melodic electronic trance track with a space and retro vibe, featuring drums, distortion guitar, flute, synth bass, and slap bass. Set in A minor with a fast tempo of 138 BPM, the song maintains a 4/4 time signature throughout its duration.",
    "A classical music piece with Jazz/Blues vibe. This music piece is influenced by George Gershwin.",
    "A classical music piece composed by Wolfgang Amadeus Mozart.",
    "This is a intermediate classical music piece. This music piece has piano in it.",
    "A rock song with strong drums and electric guitar. The tempo is very fast.",
    "This music piece is in a sad mood.",
    "A sad pop song with a strong piano presence.",
    "A soft love song on piano."
]

for text in text_container:
    for sample in range(10):
        print(f"current text = {text}, sample = {sample}")
        input_ids = tokenizer(text, 
                            return_tensors='pt', 
                            truncation=True, 
                            max_length=max_length)['input_ids']

        decoder_start_token_id = model.config.decoder_start_token_id
        eos_token_id = model.config.eos_token_id

        decoder_input_ids = torch.tensor([[decoder_start_token_id]])

        for t_idx in range(max_length):
            outputs = model(input_ids=input_ids, 
            decoder_input_ids=decoder_input_ids)
            probs = outputs.logits[0][-1]
            probs = torch.nn.Softmax(dim=-1)(probs).detach().numpy()
            sampled_id = temperature_sampling(probs=top_p_sampling(probs, 
                                                                top_p=top_p, 
                                                                return_probs=True),
                                            temperature=temperature)
            decoder_input_ids = torch.cat((decoder_input_ids, torch.tensor([[sampled_id]])), 1)
            if sampled_id!=eos_token_id:
                continue
            else:
                tune = "X:1\n"
                tune += tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)
                print(tune)
                break
        print("===================================")
