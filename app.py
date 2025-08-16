import streamlit as st
import json
import os
import io
from dotenv import load_dotenv
import pandas as pd
import altair as alt
from fpdf import FPDF

from utils.speech_to_text import transcribe_audio
from utils.speaker_diarization import diarize_audio
from utils.merge_diarization import assign_speaker_to_transcript, coalesce_segments, produce_readable_transcript
from utils.analysis import run_analysis, analyze_sentiment_per_speaker

load_dotenv()

SENTIMENT_MAP = {"Negative": 0, "Neutral": 1, "Positive": 2}

import io
from fpdf import FPDF

def generate_pdf(diarized_transcript, speaker_analysis):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Diarized Transcript & Speaker Analysis", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Diarized Transcript:", ln=True)
    pdf.set_font("Arial", "", 12)
    for seg in diarized_transcript:
        line = f"[{seg['speaker']}] {seg['start']:.2f}s - {seg['end']:.2f}s: {seg['text']}"
        line = line.replace("•", "-")  # Replace unicode bullet to ASCII
        pdf.multi_cell(0, 8, line)

    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Speaker Analysis Summaries:", ln=True)
    pdf.set_font("Arial", "", 12)
    for spk, data in speaker_analysis.items():
        pdf.cell(0, 8, f"{spk} | Sentiment: {data['sentiment']}", ln=True)
        for bullet in data["summary"].split("\n"):
            bullet = bullet.replace("•", "-")
            pdf.multi_cell(0, 8, bullet)
        pdf.ln(5)

    # Output PDF as string
    pdf_bytes = pdf.output(dest='S').encode('latin1')  # Output as byte string

    # Wrap in BytesIO for Streamlit
    return io.BytesIO(pdf_bytes)

st.title("🗣️ Audio Transcription & Speaker Analysis")

uploaded_file = st.file_uploader("Upload audio file (mp3, wav, m4a)", type=["mp3", "wav", "m4a"])
model_size = st.selectbox("Select Whisper Model Size", ["tiny", "base", "small", "medium", "large"], index=2)

if uploaded_file:
    st.audio(uploaded_file)

    # Save to temp file for processing
    with open("temp_audio_file", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Step 1: Transcription
    if st.button("Transcribe Audio"):
        st.info("Transcribing audio, please wait...")
        whisper_segments, info = transcribe_audio("temp_audio_file", model_size=model_size)
        readable_transcript = "\n".join([seg["text"] for seg in whisper_segments])
        st.subheader("Transcription Result")
        st.text_area("Transcript", readable_transcript, height=300, disabled=True)
        st.session_state["whisper_segments"] = whisper_segments
        st.session_state["language"] = str(info.language)
        st.success("Transcription completed!")

    # Step 2: Diarization (only available after transcription)
    if "whisper_segments" in st.session_state:
        if st.button("Run Speaker Diarization and Analysis"):
            st.info("Running diarization, please wait...")
            diarization_segments = diarize_audio("temp_audio_file")
            st.info("Merging diarization with transcript and coalescing segments...")
            assigned = assign_speaker_to_transcript(st.session_state["whisper_segments"], diarization_segments)
            coalesced = coalesce_segments(assigned, gap_tolerance=0.6)

            st.info("Running speaker sentiment and summary analysis...")
            analysis_results = run_analysis(coalesced)

            st.subheader("Diarized Transcript")
            readable_diarized = produce_readable_transcript(coalesced)
            st.text_area("Diarized Transcript", readable_diarized, height=300, disabled=True)

            st.subheader("Speaker Analysis")
            for spk, data in analysis_results.items():
                st.markdown(f"**{spk}** | Sentiment: {data['sentiment']}")
                st.text(data['summary'])

            # Prepare sentiment timeline chart data
            timeline_data = []
            for seg in coalesced:
                spk = seg["speaker"]
                sentiment = analyze_sentiment_per_speaker({spk: seg["text"]}).get(spk, "Unknown")
                sentiment_val = SENTIMENT_MAP.get(sentiment, None)
                if sentiment_val is not None:
                    timeline_data.append({
                        "start": seg["start"],
                        "speaker": spk,
                        "sentiment_label": sentiment,
                        "sentiment_val": sentiment_val
                    })

            df_sentiment = pd.DataFrame(timeline_data)

            if not df_sentiment.empty:
                st.subheader("Sentiment Timeline Chart")
                chart = alt.Chart(df_sentiment).mark_line(point=True).encode(
                    x=alt.X('start', title='Time (s)'),
                    y=alt.Y('sentiment_val', title='Sentiment', scale=alt.Scale(domain=[0,2]),
                            axis=alt.Axis(values=[0,1,2], labelExpr="['Negative','Neutral','Positive'][datum.value]")),
                    color='speaker',
                    tooltip=['start', 'speaker', 'sentiment_label']
                ).interactive()
                st.altair_chart(chart, use_container_width=True)

            # Download JSON
            output_json = {
                "diarized_transcript": coalesced,
                "whisper_info": {"language": st.session_state["language"]},
                "speaker_analysis": analysis_results
            }
            json_data = json.dumps(output_json, indent=2)
            st.download_button(
                label="Download Analysis JSON",
                data=json_data,
                file_name="analysis_output.json",
                mime="application/json"
            )

            # Download PDF
            pdf_bytes = generate_pdf(coalesced, analysis_results)
            st.download_button(
                label="Download Transcript & Analysis PDF",
                data=pdf_bytes,
                file_name="transcript_analysis.pdf",
                mime="application/pdf"
            )

else:
    st.info("Please upload an audio file to start.")
