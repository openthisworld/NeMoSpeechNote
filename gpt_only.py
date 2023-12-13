import os
import time
import keyboard
import openai
import requests
import numpy as np
import pyaudio
import speech_recognition as sr

# Задайте ваш ключ API від OpenAI як змінну середовища
openai.api_key = os.getenv("OPENAI_API_KEY")

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
    recognizer = sr.Recognizer()
    try:
        # Використання Google Web Speech API для розпізнавання мовлення
        transcription = recognizer.recognize_google(audio_data)
        return transcription
    except sr.UnknownValueError:
        print("Google Web Speech API could not understand the audio.")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results from Google Web Speech API; {e}")
        return ""

def generate_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

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
        response = generate_response(transcription)
        save_note(response)
        print(f"Note saved: {response}")
