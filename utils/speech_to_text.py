# utils/speech_to_text.py
from faster_whisper import WhisperModel
import torch

def transcribe_audio(audio_path, model_size="small"):
    """
    Transcribe an audio file using Faster-Whisper.

    Args:
        audio_path (str): Path to the audio file.
        model_size (str): Whisper model size ("tiny", "base", "small", "medium", "large").

    Returns:
        segments (list of dict): Each dict contains {"start": float, "end": float, "text": str}
        info (Whisper info object): Metadata including detected language.
    """

    # Auto-detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print("Loading Whisper model...")
    model = WhisperModel(model_size, device=device, compute_type="float16" if device == "cuda" else "int8")

    # Transcribe
    print("Transcribing audio...")
    segments_iter, info = model.transcribe(audio_path, beam_size=5)

    # Collect results
    segments = []
    for seg in segments_iter:
        segments.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text.strip()
        })

    print(f"Transcription done. Language: {info.language}")
    return segments, info
