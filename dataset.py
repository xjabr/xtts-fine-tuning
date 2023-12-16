"""
Build a dataset for the british south dialect from the english dialects dataset.
Dataset Format: ljspeech (See more here: https://github.com/coqui-ai/TTS/blob/dev/TTS/tts/datasets/formatters.py)
"""

import io
import os

import soundfile as sf

from pydub import AudioSegment
from datasets import load_dataset


DATASET_FOLDER = "data/british_south/"
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

def process_dataset(item):
    audio = io.BytesIO(item['audio']['bytes'])
    audio.seek(0)

    audio_info = sf.info(audio)
    line_id = item['line_id']
    frame_rate = audio_info.samplerate
    sample_width = audio_info.subtype
    channels = audio_info.channels

    print(" > Frame rate: ", frame_rate)
    print(" > Sample width: ", sample_width)
    print(" > Channels: ", channels)
    print(" > Line id: ", line_id)
    print("")

    AudioSegment.from_raw(audio, sample_width=SAMPLE_WIDTH_MAP.get(sample_width), frame_rate=frame_rate, channels=channels).export(os.path.join(WAVS_FOLDER, line_id + ".wav"), format="wav")
    METADATA_INFO.append((line_id, item['text'], item['text']))

# load the datasets
dataset = load_dataset("ylacombe/english_dialects", "southern_male", split="all")

print(" > Dataset loaded. N. of samples: ", len(dataset))
print(" > Processing dataset...\n")

# save files in the dataset folder
df = dataset.to_pandas()
df.apply(process_dataset, axis=1)

print(" > Dataset processed!")

# save the metadata
print(" > Saving metadata...")

METADATA_CSV_TEXT = "\n".join(["|".join(line) for line in METADATA_INFO])
with open(METADATA_NAME, "w") as f:
    f.write(METADATA_CSV_TEXT)

print(" > Metadata saved!")
print(" > Dataset ready on folder: ", DATASET_FOLDER)