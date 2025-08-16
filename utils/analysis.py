from collections import defaultdict
from typing import List, Dict
from transformers import pipeline
from groq import Groq
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Sentiment model (fast CPU-friendly)
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"  # latest & more accurate
)

LABEL_MAP = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

def group_text_by_speaker(coalesced_segments: List[Dict]) -> Dict[str, str]:
    """
    Returns dict: {speaker: "full concatenated text"}
    """
    grouped = defaultdict(list)
    for seg in coalesced_segments:
        grouped[seg["speaker"]].append(seg["text"])
    return {spk: " ".join(txts) for spk, txts in grouped.items()}

def analyze_sentiment_per_speaker(speaker_texts: Dict[str, str]) -> Dict[str, str]:
    """
    Returns dict: {speaker: sentiment_label} with mapped & capitalized labels.
    """
    results = {}
    for spk, text in speaker_texts.items():
        if not text.strip():
            results[spk] = "Unknown"
            continue
        snippet = text[:1000].rsplit(" ", 1)[0]
        raw_label = sentiment_analyzer(snippet)[0]["label"].strip().upper()
        mapped_label = LABEL_MAP.get(raw_label, raw_label.title())
        results[spk] = mapped_label
    return results

def summarize_per_speaker(speaker_texts: Dict[str, str], max_tokens=150):
    """
    Generates bullet-only summaries using Groq LLaMA, 
    cleaning out intros and ensuring uniform bullet style.
    """
    summaries = {}
    for spk, text in speaker_texts.items():
        if not text.strip():
            summaries[spk] = ""
            continue

        prompt = f"""
        Summarize the main points from the following text spoken by {spk}.
        Output must follow these rules exactly:
        - Only top-level bullet points starting with "• ".
        - No introduction or conclusion text at all.
        - No phrases like "Summary", "Main points", or "Here are".
        - No nested bullets, numbering, or indentation.

        Text:
        \"\"\"{text}\"\"\"
        """

        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.3
        )

        raw_summary = response.choices[0].message.content

        cleaned_lines = []
        for line in raw_summary.split("\n"):
            # Strip bullet chars and numbers, then strip whitespace
            stripped = line.strip(" \t-•0123456789.:").strip()
            if not stripped:
                continue
            # Skip lines with unwanted intro/conclusion phrases
            if any(phrase in stripped.lower() for phrase in [
                "summary", "main point", "here are", "in conclusion", "overall"
            ]):
                continue
            cleaned_lines.append(f"• {stripped}")

        summaries[spk] = "\n".join(cleaned_lines)

    return summaries

def run_analysis(coalesced_segments: List[Dict]):
    """
    Runs sentiment analysis and generates bullet-point Groq summaries
    for each speaker from the diarized transcript segments.
    """
    # Group speech per speaker
    speaker_texts = group_text_by_speaker(coalesced_segments)

    # Sentiment mapping
    sentiments = analyze_sentiment_per_speaker(speaker_texts)

    # Groq summaries (already cleaned)
    summaries = summarize_per_speaker(speaker_texts)

    results = {}
    for spk, text in speaker_texts.items():
        results[spk] = {
            "full_text": text.strip(),
            "sentiment": sentiments.get(spk, "Unknown"),  # already mapped in LABEL_MAP
            "summary": summaries[spk]
        }
    return results
