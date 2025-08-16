# utils/summarizer.py
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def summarize_and_extract_actions(transcript):
    """
    Summarizes transcript and extracts action items.
    """
    prompt = f"""
    You are an AI meeting assistant.
    1. Summarize the following meeting transcript in 5-7 bullet points.
    2. Extract clear action items with assignee and deadline if mentioned.
    
    Transcript:
    {transcript}
    """

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return response.choices[0].message.content
