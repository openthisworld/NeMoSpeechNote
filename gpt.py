import os
import time
import keyboard
import nemo
import numpy as np
import pyaudio
import requests

# Ініціалізація моделей NeMo для ASR
asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained("stt_en_quartznet15x5")

# URL та ключ для ChatGPT API
chatgpt_api_url = "https://api.openai.com/v1/chat/completions"
chatgpt_api_key = "YOUR_OPENAI_API_KEY"

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

def generate_response(prompt):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {chatgpt_api_key}"
    }
    data = {
        "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
    }
    response = requests.post(chatgpt_api_url, headers=headers, json=data)
    return response.json()["choices"][0]["message"]["content"]

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
