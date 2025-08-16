# utils/speaker_diarization.py
import os
from pyannote.audio import Pipeline
from dotenv import load_dotenv

load_dotenv()

def diarize_audio(audio_path):
    """
    Returns a list of diarization segments:
      [{"start": float, "end": float, "speaker": "SPEAKER_00"}, ...]
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("Please set HF_TOKEN in .env")

    print("Loading pyannote speaker-diarization pipeline...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)

    print("Running diarization...")
    diarization = pipeline(audio_path)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": speaker
        })

    return segments
