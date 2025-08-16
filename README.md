# AI-Powered Meeting Notes & Action Items Extractor

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-v1.0+-orange)](https://streamlit.io/)

---

## 🚀 Project Overview

This project is an AI-powered meeting assistant that takes meeting audio recordings as input and produces:

- **Accurate transcriptions** using Whisper speech-to-text.
- **Speaker diarization** to identify and segment speakers.
- **Summarization** into clear, bullet-point key points using Groq API (LLaMA 3.1 or Mixtral).
- **Action item extraction** with deadlines and assignments (planned feature).
- **Interactive UI** built with Streamlit to upload audio, view transcripts, summaries, and download reports.
- **Exportable PDF/JSON** reports summarizing meeting content and speaker analysis.

---

## 🎯 Features

- Upload or record meeting audio (.mp3, .wav, .m4a).
- Fast, local transcription with Whisper (supports GPU acceleration).
- Speaker diarization for multiple participant meetings.
- AI-generated bullet-point summaries using LLaMA 3.1 via Groq API.
- Sentiment analysis per speaker.
- Clean, interactive web UI with transcript, summaries, sentiment charts.
- Downloadable PDF and JSON reports.
- Modular, testable backend utilities for each pipeline step.

---

## 🛠️ Tech Stack

| Component                 | Technology / Library                 |
| -------------------------| ---------------------------------- |
| Speech-to-Text           | [Whisper](https://github.com/openai/whisper), [faster-whisper](https://github.com/guillaumekln/faster-whisper) |
| Speaker Diarization      | pyannote.audio / custom algorithms  |
| Summarization & Analysis | Groq API (LLaMA 3.1 / Mixtral)      |
| UI                       | [Streamlit](https://streamlit.io/) |
| PDF Generation           | [FPDF](https://pyfpdf.github.io/)   |
| Python Environment       | 3.8+                               |

---

## 🧩 Project Structure

```plaintext
meeting-notes-ai/
│
├── app.py                     # Streamlit application
├── requirements.txt           # Dependencies
├── README.md                  # This file
├── .env.example               # Sample environment variables file
├── utils/                     # Modular backend utils
│   ├── speech_to_text.py
│   ├── speaker_diarization.py
│   ├── merge_diarization.py
│   ├── analysis.py
├── tests/                     # Unit tests for utils
├── sample_audio/              # Example audio files
└── docs/                      # Documentation & diagrams
```

---

## ⚡ Quick Start

**Clone the repository**

```bash
git clone https://github.com/DeveshPant18/meeting-notes-ai.git
cd meeting-notes-ai
```

**Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate    # On Windows use `venv\Scripts\activate`
```

**Install dependencies**

```bash
pip install -r requirements.txt
```

**Set environment variables**

Copy `.env.example` to `.env` and update it with your secrets:

```bash
cp .env.example .env
```

Add your **Groq API key** and other configuration variables.

**Run the Streamlit app**

```bash
streamlit run app.py
```

Upload an audio file and start analysis!

---

## 🧪 Testing

Run tests with:

```bash
pytest tests/
```

---

## 📈 Future Enhancements

- Add action item extraction with deadlines and responsible person assignment.
- Support direct audio recording from the UI.
- Improve diarization accuracy with custom models.
- Add multi-language transcription and summarization.
- Deploy on cloud with GPU acceleration.

---

## 🙌 Contributing

Contributions, issues, and feature requests are welcome!

Please follow contributing guidelines (if any) and code of conduct.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📞 Contact

**Devesh Pant**  
Email: [ma24m007@smail.iitm.ac.in](mailto:ma24m007@smail.iitm.ac.in)  
GitHub: [github.com/DeveshPant18](https://github.com/DeveshPant18)

Built with ❤️ for IIT Madras Industrial Mathematics & Scientific Computing M.Tech project
