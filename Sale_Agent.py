import asyncio
import json
import os
import time
import wave
from io import BytesIO

import edge_tts
import groq
import numpy as np
import pyaudio
import speech_recognition as sr
import sounddevice as sd
import soundfile as sf
import torch

GROQ_API_KEY = "apni key add kro"
VOICE = "en-US-AriaNeural"
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_SIZE = 512
SILENCE_THRESHOLD = 300  
SILENCE_DURATION = 1.0  

# VAD Parameters
VAD_THRESHOLD = 0.5  
SPEECH_PAD_MS = 300  

# Initialize clients
groq_client = groq.Client(api_key=GROQ_API_KEY)

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Initialize speech recognizer
recognizer = sr.Recognizer()
recognizer.energy_threshold = 300  

# Flag for interruption detection
speaking = False
should_stop_speaking = False

# Initialize Silero VAD
vad_model = None


def init_vad_model():
    global vad_model
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False
    )
    vad_model = model
    return model, utils


def is_speech(audio_chunk, vad_model, threshold=VAD_THRESHOLD):
    audio_tensor = torch.FloatTensor(np.frombuffer(audio_chunk, dtype=np.int16).copy() / 32768.0)
    speech_prob = vad_model(audio_tensor, RATE).item()
    return speech_prob >= threshold


def play_audio(audio_data, rate=24000):
    global speaking
    speaking = True
    try:
        sd.play(audio_data, rate)
        sd.wait()
    except:
        pass
    finally:
        speaking = False


async def listen_for_interruption(stop_event):
    global should_stop_speaking, vad_model

    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )

    consecutive_speech_chunks = 0
    try:
        while not stop_event.is_set() and speaking:
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            if is_speech(data, vad_model, threshold=0.8): 
                consecutive_speech_chunks += 1
            else:
                consecutive_speech_chunks = 0
            if consecutive_speech_chunks >= 2:  
                should_stop_speaking = True
                print("User interruption detected with VAD!")
                break

            await asyncio.sleep(0.01)
    finally:
        stream.stop_stream()
        stream.close()


async def recognize_speech():
    global vad_model

    print("Listening... (speak now)")
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )
    frames = []
    speech_detected = False
    speech_start_time = None
    silent_chunks = 0
    max_silent_chunks = int(SILENCE_DURATION * RATE / CHUNK_SIZE)

    try:
        while True:
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            frames.append(data)

            # Use Silero VAD to detect speech
            current_is_speech = is_speech(data, vad_model)

            if current_is_speech:
                if not speech_detected:
                    speech_detected = True
                    speech_start_time = time.time()
                silent_chunks = 0
            elif speech_detected:
                silent_chunks += 1

            # End recording if enough silence after speech
            if speech_detected and silent_chunks >= max_silent_chunks:
                print(f"Speech ended, silence detected for {SILENCE_DURATION}s")
                break

            # Timeout if recording too long
            if len(frames) > 750:  # ~15 seconds max
                print("Maximum recording length reached")
                break
    finally:
        stream.stop_stream()
        stream.close()

    if not speech_detected:
        print("No speech detected")
        return None

    print("Processing speech...")
    wav_buffer = BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    wav_buffer.seek(0)
    start_time = time.time()
    try:
        with sr.AudioFile(wav_buffer) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            elapsed = time.time() - start_time
            print(f"User: {text}")
            print(f"Speech recognition took {elapsed:.2f} seconds")
            return text
    except sr.UnknownValueError:
        print("Could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"Speech recognition error: {e}")
        return None


async def get_llm_response(text):
    try:
        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system",
                 "content": "You are a concise, friendly sales agent. Keep responses under 30 words. Focus on selling your product with brief, compelling points."},
                {"role": "user", "content": text}
            ],
            max_tokens=100,  
            temperature=0.5,
            stream=True
        )

        full_response = ""
        for chunk in response:
            if should_stop_speaking:
                break

            if chunk.choices[0].delta.content:
                chunk_text = chunk.choices[0].delta.content
                full_response += chunk_text
                print(f"{chunk_text}", end="", flush=True)

        print()  
        return full_response
    except Exception as e:
        print(f"Error with Groq: {e}")
        return "Sorry, I encountered an error while processing your request."


async def text_to_speech(text):
    global should_stop_speaking
    should_stop_speaking = False

    try:
        temp_filename = "temp_speech.mp3"
        communicate = edge_tts.Communicate(text, VOICE)
        await communicate.save(temp_filename)
        stop_event = asyncio.Event()
        interruption_task = asyncio.create_task(listen_for_interruption(stop_event))

        audio_data, sample_rate = sf.read(temp_filename)
        play_audio(audio_data, sample_rate)
        stop_event.set()
        await interruption_task
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    except Exception as e:
        print(f"🚫 TTS error: {e}")


async def sales_greeting():
    greeting = "Hi there! I'm your personal tech assistant. I'd love to tell you about our new AI-powered smartphone. How can I help you today?"
    await text_to_speech(greeting)


async def main():
    print("Voice agent is running! Press Ctrl+C to exit.")
    global vad_model
    print("Loading Silero VAD model...")
    vad_model, _ = init_vad_model()
    print("Silero VAD model loaded")

    # Start with sales greeting
    await sales_greeting()

    try:
        while True:
            global should_stop_speaking
            if should_stop_speaking:
                should_stop_speaking = False
            user_text = await recognize_speech()

            if user_text:
                llm_response = await get_llm_response(user_text)
                if llm_response and not should_stop_speaking:
                    await text_to_speech(llm_response)

            await asyncio.sleep(0.1)

    except KeyboardInterrupt:
        print("Voice agent stopped.")
    finally:
        audio.terminate()
if __name__ == "__main__":
    asyncio.run(main())