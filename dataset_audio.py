"""
Build a dataset for the british south dialect from the english dialects dataset.
Dataset Format: ljspeech (See more here: https://github.com/coqui-ai/TTS/blob/dev/TTS/tts/datasets/formatters.py)
"""

import io
import os
import sys
import torch
import soundfile as sf
import moviepy.editor as mpe

from pydub import AudioSegment
from datasets import load_dataset
import whisper_timestamped as whisper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_FOLDER = "data/stephen-fry/"
WAVS_FOLDER = os.path.join(DATASET_FOLDER, "wavs/")
METADATA_NAME = os.path.join(DATASET_FOLDER, "metadata.csv")
SAMPLE_WIDTH_MAP = {
    'PCM_16': 2,
    'PCM_24': 3,
    'PCM_32': 4,
    'FLOAT': 4,
    'DOUBLE': 8
}
METADATA_INFO = []

# build the folders
os.makedirs(DATASET_FOLDER, exist_ok=True)
os.makedirs(WAVS_FOLDER, exist_ok=True)

print(" > Dataset folder: ", DATASET_FOLDER)
print(" > Wavs folder: ", WAVS_FOLDER)
print(" > Metadata file: ", METADATA_NAME)

# file audio path
file_audio = sys.argv[1]

# loading whisper
print(" > Loading whisper...")
model = whisper.load_model("large", device=device)

print(" > Loading audio...", file_audio)
audio = whisper.load_audio(file_audio)

print(" > Generating timestamps...")
transcript = whisper.transcribe(model, audio, language="en")

# get segments
BATCH_SIZE = 64

segments = []
for i in range(0, len(transcript['segments']), BATCH_SIZE):
    tmp = transcript['segments'][i:i + BATCH_SIZE]
    
    result = []
    for t in tmp:
        result.append((t["text"], t["start"], t["end"]))
            
    segments += result


print(" > Saving segments...")

for segment in segments:
    filename_audio = f"AUDIO_WAV_{segment[1]}_{segment[2]}"
    sub_audio = mpe.AudioFileClip(file_audio).subclip(segment[1], segment[2]).write_audiofile(os.path.join(WAVS_FOLDER, f"{filename_audio}.wav"))
    METADATA_INFO.append((filename_audio, segment[0], segment[0]))

# save the metadata
print(" > Saving metadata...")

METADATA_CSV_TEXT = "\n".join(["|".join(line) for line in METADATA_INFO])
with open(METADATA_NAME, "w") as f:
    f.write(METADATA_CSV_TEXT)

print(" > Dataset ready on folder: ", DATASET_FOLDER)