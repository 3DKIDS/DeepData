import cv2
import base64
import time
from openai import OpenAI
import os
import requests
import io
from pydub import AudioSegment
from pydub.playback import play
import threading
from pathlib import Path
import pygame

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "당신의키"))

def capture_frames(num_frames=5, interval=1):
    cap = cv2.VideoCapture(1)  # 0은 기본 USB 카메라를 의미합니다
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        time.sleep(interval)
    cap.release()
    return frames

def encode_frames(frames):
    return [base64.b64encode(cv2.imencode(".jpg", frame)[1]).decode("utf-8") for frame in frames]

def analyze_frames(encoded_frames):
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                "These are frames from a live USB camera feed. Describe korean what you see in detail. Include any notable objects, people, or activities.",
                *map(lambda x: {"image": x, "resize": 768}, encoded_frames),
            ],
        },
    ]
    params = {
        "model": "gpt-4o",
        "messages": PROMPT_MESSAGES,
        "max_tokens": 1000,
    }

    result = client.chat.completions.create(**params)
    return result.choices[0].message.content

def text_to_speech(text):
    speech_file_path = Path(__file__).parent / "speech.mp3"
    with client.audio.speech.with_streaming_response.create(
    model="tts-1",
    voice="alloy",
    input=text
    ) as response:
        response.stream_to_file(speech_file_path)

    return speech_file_path

def play_audio(audio_data):
    #audio play init
    music_file = audio_data   # mp3 or mid file
    freq = 16000    # sampling rate, 44100(CD), 16000(Naver TTS), 24000(google TTS)
    bitsize = -16   # signed 16 bit. support 8,-8,16,-16
    channels = 1    # 1 is mono, 2 is stereo
    buffer = 2048   # number of samples (experiment to get right sound)
    pygame.mixer.init(freq, bitsize, channels, buffer)
    pygame.mixer.music.load(music_file)
    pygame.mixer.music.play()
    clock = pygame.time.Clock()
    while pygame.mixer.music.get_busy():
             clock.tick(30)
    pygame.mixer.quit()

def display_video(stop_event):
    cap = cv2.VideoCapture(0)
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            cv2.imshow("USB Camera Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

def main():
    stop_event = threading.Event()
    video_thread = threading.Thread(target=display_video, args=(stop_event,))
    video_thread.start()

    try:
        while True:
            print("Capturing frames...")
            frames = capture_frames()
            encoded_frames = encode_frames(frames)
            
            print("Analyzing frames...")
            description = analyze_frames(encoded_frames)
            print("\nScene Description:")
            print(description)
            
            print("\nGenerating audio...")
            audio_data = text_to_speech(description)
            
            print("Playing audio description...")
            play_audio(audio_data)
            
            user_input = input("\nPress Enter to continue or type 'q' to quit: ")
            if user_input.lower() == 'q':
                break
    finally:
        stop_event.set()
        video_thread.join()

if __name__ == "__main__":
    main()
