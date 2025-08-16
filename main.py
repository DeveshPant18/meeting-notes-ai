# main.py
import os
import json
from dotenv import load_dotenv
from utils.speech_to_text import transcribe_audio
from utils.speaker_diarization import diarize_audio
from utils.merge_diarization import (
    assign_speaker_to_transcript,
    coalesce_segments,
    produce_readable_transcript
)
from utils.analysis import run_analysis  # Handles sentiment mapping + Groq summaries

# Load environment variables (for GROQ_API_KEY, etc.)
load_dotenv()

if __name__ == "__main__":
    audio_file = "sample_audio/meeting.mp3"

    # Step 1 – Transcribe
    whisper_segments, info = transcribe_audio(audio_file, model_size="small")

    # Step 2 – Diarize
    diarization_segments = diarize_audio(audio_file)
    assigned = assign_speaker_to_transcript(whisper_segments, diarization_segments)
    coalesced = coalesce_segments(assigned, gap_tolerance=0.6)

    # Step 3 – Pretty merged transcript
    readable = produce_readable_transcript(coalesced)
    print("\n=== Diarized Transcript ===\n")
    print(readable)

    # Step 4 – Sentiment + Groq bullet summaries in one go
    analysis_results = run_analysis(coalesced)

    # Step 5 – Print speaker analysis
    print("\n=== Speaker Analysis ===\n")
    for spk, data in analysis_results.items():
        print(f"{spk} | Sentiment: {data['sentiment']}")
        print(data["summary"] + "\n")

    # Step 6 – Save JSON output
    output_data = {
        "readable_transcript": readable,  # now a clean merged text transcript
        "diarized_segments": coalesced,
        "whisper_info": {"language": str(info.language)},
        "speaker_analysis": analysis_results
    }
    os.makedirs("output", exist_ok=True)
    with open("output/diarized_transcript_with_analysis.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("\nSaved: output/diarized_transcript_with_analysis.json")
