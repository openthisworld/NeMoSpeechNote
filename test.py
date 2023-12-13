import os
import time
import keyboard
import nemo
import numpy as np
import pyaudio
from nemo.collections import nlp, asr

# Ініціалізація моделей NeMo для ASR та NLP
asr_model = asr.models.EncDecCTCModel.from_pretrained("stt_en_quartznet15x5")
nlp_model = nlp.models.PunctuationCapitalizationModel.from_pretrained("punctuation_capitalization")

# Параметри для запису аудіо
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    frames = []
    print("Recording... (Press Space to stop)")

    while True:
        if keyboard.is_pressed(' '):
            break
        data = stream.read(CHUNK)
        frames.append(np.frombuffer(data, dtype=np.int16))

    print("Recording stopped.")
    
    stream.stop_stream()
    stream.close()
    p.terminate()

    return np.concatenate(frames, axis=0)

def transcribe_audio(audio_data):
    transcriptions = asr_model.transcribe([audio_data])
    return transcriptions[0]

def analyze_text(transcription):
    analyzed_text = nlp_model.add_punctuation_capitalization([transcription])
    return analyzed_text[0]

def save_note(note):
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"note_{timestamp}.txt"
    
    with open(filename, 'w') as file:
        file.write(note)

if __name__ == "__main__":
    while True:
        input("Press Enter to start recording...")
        audio_data = record_audio()
        transcription = transcribe_audio(audio_data)
        analyzed_text = analyze_text(transcription)
        save_note(analyzed_text)
        print(f"Note saved: {analyzed_text}")
