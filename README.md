# NoteBot

NoteBot is an AI-powered application that helps students convert audio recordings to text and generate summaries. It uses Whisper for speech-to-text conversion and Gemini for text summarization.

## Features
- Upload audio files (.mp3, .wav)
- Convert speech to text using Whisper
- Generate AI-powered summaries using Gemini

## Setup Instructions

1. Create a virtual environment and activate it:
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory and add your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

4. Run the application:
```bash
streamlit run app.py
```

## Supported File Formats
- MP3 (.mp3)
- WAV (.wav)

## Note
Make sure you have a valid Google API key for the Gemini model to work properly. 