<p align="center">
<strong><a href="https://wx83.github.io/">Weihan Xu</a></strong> <sup>1</sup> &emsp;
<strong><a href="https://cseweb.ucsd.edu/~jmcauley/">Julian McAuley</a></strong> <sup>2</sup> &emsp;
<strong><a href="https://cseweb.ucsd.edu/~tberg/">Taylor Berg-Kirkpatrick</a></strong> <sup>2</sup> &emsp;
<strong><a href="https://music-cms.ucsd.edu/people/faculty/regular_faculty/shlomo-dubnov/index.html">Shlomo Dubnov</a></strong> <sup>2</sup> &emsp;
<strong><a href="https://salu133445.github.io/">Hao-Wen Dong</a></strong> <sup>2,3</sup><br>
<sup>1</sup> Duke University &emsp;<sup>2</sup> University of California San Diego &emsp;<sup>3</sup> University of Michigan
</p>

<p align="center">
<strong><a href="https://github.com/wx83/MetaScore_Official/tree/codebase"> [Codebase] </a></strong> &emsp;
<strong><a href="https://arxiv.org/pdf/2410.02084"> [Paper] </a></strong>&emsp;
<strong><a href="https://zenodo.org/records/17290490?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjE5NzhkMDUzLTAzYjQtNDEwNC1iYTQxLTk3ZGI4MTFjOGRmYSIsImRhdGEiOnt9LCJyYW5kb20iOiIzNzJkYmI1YTFhODQ4OTVlOTExZTY0ODFlNDljYmUzZCJ9.MXJU146ZPly6kecnzKQ-CMoa0XHO6ZSdrKca5U0fCiO8WQVyxXEvVLuu49Nn4r4t33SkFb4Y2bNDGKZYGSYCiw"> [Data] </a></strong>&emsp;
</p>

# Content
1. Selected Tag- and Text-Conditioned Generation Examples
2. Summary of MetaScore Dataset
3. Summary of Conditioned Music Generation Models
4. MetaScore Quantative Examples
5. MetaScore Annotation Examples
6. LLM Template
7. More Examples (Unselected)

---

# Selected Tag- and Text-Conditioned Generated Music
> Settings: In terms of tag-conditioned music generation, we initiate with a "start-of-song" event, then feed relevant tags into either MST-Tags-Small or MST-Tags. This is followed by a "start-of-notes" event to allow the model to begin generating notes. For text-conditioned music generation, we input the text embedding corresponding to the provided text into MST-Text and use a "start-of-song" event to prompt the model to start generating notes.


## I. Multiple Control: 
<div style="overflow-x: auto;" markdown="block">

| Model | Condition | Audio Sample
|--------|----------|--------|
|MST-Tags-Small| Classical/Traditional, Hard, ludwig van beethoven, Piano| {% include audio_player.html filename="audio/demopage/tag_highq/hard_beethoven_cond.wav" %} 
|MST-Tags | Classical/Traditional, Hard, ludwig van beethoven, Piano | {% include audio_player.html filename="audio/demopage/tag_all/hard_beethoven_cond.wav" %} 
|MST-Text | This is an easy classical piano piece composed by wolfgang amadeus mozart.|{% include audio_player.html filename="audio/demopage/text/easymozart.wav" %}

</div>

## II. Multiple Tracks: 
<div style="overflow-x: auto;" markdown="block">

| Model | Condition | Audio Sample
|--------|----------|--------|
|MST-Tags-Small|Rock/Metal, Electric-guitar, Electric-bass, Piano|{% include audio_player.html filename="audio/demopage/tag_highq/multitrack_cond.wav" %} 
|MST-Tags|Rock/Metal, Electric-guitar, Electric-bass, Piano|{% include audio_player.html filename="audio/demopage/tag_all/multitrack_cond.wav" %} 
| MST-Text|This is a rock or metal music. It has piano, guitar and bass within it.|{% include audio_player.html filename="audio/demopage/text/multitrack_cond.wav" %} 

</div>


## III. Drum Accompaniment:

<div style="overflow-x: auto;" markdown="block">

| Model | Condition | Audio Sample
|--------|----------|--------|
|MST-Tags-Small| Drum, Synth_Bass|{% include audio_player.html filename="audio/demopage/tag_highq/drum_synth_bass_conditioned.wav" %} 
|MST-Tags|Drum, Contrabass, Piccolo|{% include audio_player.html filename="audio/demopage/tag_all/drum_cond.wav" %} 
|MST-Text|This music has drum within it|{% include audio_player.html filename="audio/demopage/text/drum_conditioned.wav" %} 

</div>

## IV: Free-form User-Annotated Tags:
<div style="overflow-x: auto;" markdown="block">

| Model | Condition | Audio Sample
|--------|----------|--------|
|MST-Text|A short and emotional music piece inspired by an anime scene.|{% include audio_player.html filename="audio/useranno/anime.wav" %} 
|MST-Text|A powerful orchestral music piece.|{% include audio_player.html filename="audio/useranno/orchestra.wav" %} 
|MST-Text|An upbeat background music piece for a fun video game.|{% include audio_player.html filename="audio/useranno/videogame.wav" %} 

</div>
---

# Summary of MetaScore Dataset
<div style="overflow-x: auto;" markdown="block">

|Dataset|Description|Number of Samples|Tag Information| Text Information|
|--------|----------|--------|--------|--------|
|MetaScore-Raw| The raw MuseScore files and metadata scraped from the MuseScore forum| 963K| ✔||
| MetaScore-Genre| A subset of MetaScore-Full that contains ground truth genre tags and LLM-generated Captions | 181K | ✔| ✔|
| MetaScore-Plus| MetaScore-Raw were missing genre tags are completed by the trained genre tagger. LLM captions based on those tags are also provided| 963K|  ✔|✔ |

</div>


# Summary of Conditioned Music Generation Models
<div style="overflow-x: auto;" markdown="block">

|Model| Training Dataset| Input Type| Model Size| Training Samples|
|--------|----------|--------|--------|--------|
|MST-Tags-Small| MetaScore-Genre| Tag| 87.36M| 150K|
|MST-Tags| MetaScore-Plus| Tag| 87.36M| 901K|
|MST-Text| MetaScore-Plus| Text| 87.44M| 560K|

</div>

---

# MetaScore Quantative Examples:
> 
To evaluate the quality of our dataset, we used the rating entry (MetaScore-Raw) as an indicator. This rating, which ranges from 1 to 5 (with 5 being the highest), serves as a structured measure of perceived music quality. To demonstrate that even lower-rated entries can still be suitable for use, we randomly selected 10 samples from three categories: low ratings (below 3 or not rated), mid-range ratings (3–4), and high ratings (above 4). These examples illustrate the overall usability and diversity of the dataset across different rating levels.


<div style="overflow-x: auto;" markdown="block">
<style>
    audio {
    max-width: 250px;
    }
</style>



|Rate | Audio Sample |
|--------|----------|
| 4 ~ 5 | {% include audio_player.html filename="audio/highrate/0_truth.wav" %} | 
| 4 ~ 5  | {% include audio_player.html filename="audio/highrate/1_truth.wav" %} |
| 4 ~ 5  | {% include audio_player.html filename="audio/highrate/2_truth.wav" %} |
| 4 ~ 5    | {% include audio_player.html filename="audio/highrate/3_truth.wav" %} |
| 4 ~ 5    | {% include audio_player.html filename="audio/highrate/4_truth.wav" %} |
| 4 ~ 5    | {% include audio_player.html filename="audio/highrate/5_truth.wav" %} |
|  4 ~ 5   | {% include audio_player.html filename="audio/highrate/6_truth.wav" %} | 
|  4 ~ 5   | {% include audio_player.html filename="audio/highrate/7_truth.wav" %} |
| 4 ~ 5    | {% include audio_player.html filename="audio/highrate/8_truth.wav" %} |
|  4 ~ 5   | {% include audio_player.html filename="audio/highrate/9_truth.wav" %} |

|Rate | Audio Sample |
|--------|----------|
|  3 ~ 4| {% include audio_player.html filename="audio/midrate/0_truth.wav" %} | 
| 3 ~ 4  | {% include audio_player.html filename="audio/midrate/1_truth.wav" %} |
| 3 ~ 4  | {% include audio_player.html filename="audio/midrate/2_truth.wav" %} |
| 3 ~ 4  | {% include audio_player.html filename="audio/midrate/3_truth.wav" %} |
| 3 ~ 4   | {% include audio_player.html filename="audio/midrate/4_truth.wav" %} |
| 3 ~ 4    | {% include audio_player.html filename="audio/midrate/5_truth.wav" %} |
|  3 ~ 4   | {% include audio_player.html filename="audio/midrate/6_truth.wav" %} | 
|  3 ~ 4  | {% include audio_player.html filename="audio/midrate/7_truth.wav" %} |
| 3 ~ 4  | {% include audio_player.html filename="audio/midrate/8_truth.wav" %} |
|  3 ~ 4  | {% include audio_player.html filename="audio/midrate/9_truth.wav" %} |

|Rate | Audio Sample |
|--------|----------|
|  < 3| {% include audio_player.html filename="audio/lowrate/0_truth.wav" %} | 
| < 3  | {% include audio_player.html filename="audio/lowrate/1_truth.wav" %} |
| < 3 | {% include audio_player.html filename="audio/lowrate/2_truth.wav" %} |
| < 3 | {% include audio_player.html filename="audio/lowrate/3_truth.wav" %} |
| < 3 | {% include audio_player.html filename="audio/lowrate/4_truth.wav" %} |
|  < 3  | {% include audio_player.html filename="audio/lowrate/5_truth.wav" %} |
| < 3 | {% include audio_player.html filename="audio/lowrate/6_truth.wav" %} | 
| < 3 | {% include audio_player.html filename="audio/lowrate/7_truth.wav" %} |
| < 3 | {% include audio_player.html filename="audio/lowrate/8_truth.wav" %} |
|  < 3 | {% include audio_player.html filename="audio/lowrate/9_truth.wav" %} |


---

# MetaScore Annotation Examples
> We show annotation examples in MetaScore in the following table. The first two examples are from genre tagger test set with both true genre tags and inferred genre tags. Example 3,4, 6 and 7 are from those without true genre tags. Example 5 doesn't have inferred genre tags because they are from genre tagger training set. *Note that the following samples are not generated by AI.*

<div style="overflow-x: auto;" markdown="block">
<style>
    audio {
    max-width: 250px;
    }
</style>

|Audio Sample | True Genre | Inferred Genre |Extracted Metadata| LLM-Captions
|--------|----------|--------|--------|--------|
|{% include audio_player.html filename="audio/demopage/annotation/0_truth.wav" %} | Classical/Traditional| Classical/Traditional| Classical/Traditional; Easy; William Marshall; 1 view| A easy classical music piece composed by William Marshall.
|{% include audio_player.html filename="audio/demopage/annotation/1_truth.wav" %}| Rock/Metal | Rock/Metal | Rock/Metal; Intermediate; Piano; 5 comments; 158 favorites; 5862 views| A music piece with a rock/metal vibe.
|{% include audio_player.html filename="audio/demopage/annotation/2_truth.wav" %} | ✘ | Soundtrack/Stage | Sountrack/Stage; Trombone, Piano | This is a soundtrack with a trombone and piano in it
|{% include audio_player.html filename="audio/demopage/annotation/3_truth.wav" %} | ✘ | Jazz/Blues| Jazz/Blues; intermediate;Piano; 1 comment, 6 favorites and 462 views| A intermediate piano music piece. This music has jazz/blues vibe. 
|{% include audio_player.html filename="audio/demopage/annotation/4_truth.wav" %}  | Rock/Metal | ✘ | Rock/Metal; Bass; 2 comments; 255 favorites; 5670 views; acaphella; bad romance; cover; lady gaga | A bass music piece with rock/netal vibe. This music piece is a cover of Lady Gaga's Bad Romance. 
|{% include audio_player.html filename="audio/llm_annotate/QmS2AjwX35Ynz27Tag4tBad5fCRk9GN1tVor9tAURM855E.wav" %} |  ✘ | Rock/Metal |Rock/Metal; intermediate; Comp: JJ Lin", "Arr: Daniel Cheah; 4/4 time, Piano, 2 comments, 200 favorites; 3662 vides; 80 tempo | A piano piece in the style of JJ Lin, in 4/4 time, with a tempo of 80 bpm
|{% include audio_player.html filename="audio/llm_annotate/QmS2mNSJzE8poW3M5vqdJAB9wD9nXcLgJe8CNJ18ecjKbD.wav" %} |  ✘ | Folk/Country |Folk/Country; easy; D major; 4/4 time; 2 views| A simple 4/4 piece in D major

**Note: LLM-captions are generated using true genre tags when these are available. If true tags are not available, the captions are generated using inferred genre tags instead.**

</div>

---

# LLM Template

<img title="LLM Template" alt="Alt text" src="newquery.png">


---
# More Examples:

## I: Examples in MST-Tags-Small(Unselected)
> The following music pieces are generated by MST-Tags-Small given tag conditions.

<div style="overflow-x: auto;" markdown="block">

|Condition | Audio Sample|
|--------|----------|
|Easy, Piano|{% include audio_player.html filename="audio/highq/difficulty/easy/5_conditioned.wav" %} 
|Advanced, Piano|{% include audio_player.html filename="audio/highq/difficulty/hard/3_conditioned.wav" %} 
|Jazz/Blues, Piano, tenor-saxophone|{% include audio_player.html filename="audio/highq/40_conditioned.wav" %} 
|Classical/Traditional, Wolfgang Amadeus Mozart, Easy, Piano|{% include audio_player.html filename="audio/highq/MC/0_conditioned.wav" %} 
|Electronic/Dance|{% include audio_player.html filename="audio/highq/RC/ED/1_conditioned.wav" %}

</div>

## II: Examples in MST-Tags(Unselected)
> The following music pieces are generated by MST-Tags given tag conditions.

<div style="overflow-x: auto;" markdown="block">

|Condition | Audio Sample|
|--------|----------|
|Easy, Piano|{% include audio_player.html filename="audio/whole/difficulty/easy/1_conditioned.wav" %} 
|Advanced, Piano|{% include audio_player.html filename="audio/whole/difficulty/hard/12_conditioned.wav" %} 
|Classical/Traditional, Wolfgang Amadeus Mozart, Easy, Piano|{% include audio_player.html filename="audio/whole/multitag/2_conditioned.wav" %} 
|Electronic/Dance|{% include audio_player.html filename="audio/whole/RC/electronic/4_conditioned.wav" %} 
|Jazz/Blues|{% include audio_player.html filename="audio/whole/RC/jazz/9_conditioned.wav" %}

</div>

## III: Examples in MST-Text and BART-base(Unselected)
> The following music pieces are generated by MST-Text or BART-base Text-to-Music given text prompts. In each row, we present one example generated with MST-Text and another with BART-base Text-to-Music, both generated from the same text prompt.

<div style="overflow-x: auto;" markdown="block">

|Text Prompts | Audio Sample |Audio Sample 
|             |  Generated with MST-Text |  Generated with BART-base
|--------|----------|----------|
|This music piece is in a sad mood.|{% include audio_player.html filename="audio/Text/sadmood.wav" %} |{% include audio_player.html filename="audio/HuggingfaceWAV/sad.wav" %} 
|A classical music piece influenced by Wolfgang Amadeus Mozart.|{% include audio_player.html filename="audio/Text/QmS5zFoSSppb8Buu5saZ8X5Kyjea6fPdBKZS3WXFwLmjZR-gen.wav" %} |{% include audio_player.html filename="audio/HuggingfaceWAV/2.wav" %} 
| This is a soundtrack with electronic/dance vibe. |{% include audio_player.html filename="audio/Text/Qma6qjPQyzNCEiwA9MFFnynPthDMSTo2Bx3zPCQSJDfnTk-gen.wav" %} |{% include audio_player.html filename="audio/HuggingfaceWAV/5.wav" %} 
| This is an easy folk/country music piece.|{% include audio_player.html filename="audio/Text/country_music.wav" %}|{% include audio_player.html filename="audio/HuggingfaceWAV/1.wav" %}
|This is a intermediate classical music piece. This music piece has piano in it. |{% include audio_player.html filename="audio/Text/intermediate.wav" %} |{% include audio_player.html filename="audio/HuggingfaceWAV/4.wav" %} 

## IV: Failure Case:

<div style="overflow-x: auto;" markdown="block">

|Text Prompts | Audio Sample |Audio Sample 
|             |  Generated with MST-Text |  Generated with BART-base
|--------|----------|----------|
| A classical music piece with Jazz/Blues vibe. This music piece is influenced by George Gershwin. |{% include audio_player.html filename="audio/Text/QmdGRjfb9hiYqxCf588FHAsuPCQD3tqT544L4aonihccwj-gen.wav" %} |{% include audio_player.html filename="audio/HuggingfaceWAV/3.wav" %} 
|This is a cozy music piece.|{% include audio_player.html filename="audio/Text/cozy.wav" %} |{% include audio_player.html filename="audio/HuggingfaceWAV/cozy.wav" %} 
|A classical music piece composed by Michael Jackson. |{% include audio_player.html filename="audio/Text/mjpiano.wav" %} |{% include audio_player.html filename="audio/HuggingfaceWAV/mjpiano.wav" %} 

</div>

