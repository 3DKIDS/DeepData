import pyaudio
import wave
from pydub import AudioSegment
import keyboard
import io
import whisper
import os
from openai import OpenAI
from pathlib import Path
import pygame

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "당신의키"))

def select_microphone():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
    
    device_index = int(input("사용할 입력 장치 번호를 입력하세요: "))
    p.terminate()
    return device_index

def record_audio(filename="input.mp3", sample_rate=44100, channels=1, chunk=1024, device_index=None):
    audio_format = pyaudio.paInt16
    p = pyaudio.PyAudio()

    print("녹음을 시작합니다. 'S' 키를 누르면 녹음이 종료됩니다.")
    
    stream = p.open(format=audio_format,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=chunk)

    frames = []
    recording = True

    def stop_recording():
        nonlocal recording
        recording = False

    keyboard.on_press_key("s", lambda _: stop_recording())

    while recording:
        data = stream.read(chunk)
        frames.append(data)

    print("녹음이 완료되었습니다.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wav_buffer = io.BytesIO()
    wf = wave.open(wav_buffer, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(audio_format))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    wav_buffer.seek(0)
    sound = AudioSegment.from_wav(wav_buffer)
    sound.export(filename, format="mp3")

    print(f"오디오가 {filename}로 저장되었습니다.")
    return filename

def transcribe_audio(audio_file):
    model = whisper.load_model("base")
    
    audio = whisper.load_audio(audio_file)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    _, probs = model.detect_language(mel)
    detected_language = max(probs, key=probs.get)
    print(f"감지된 언어: {detected_language}")

    options = whisper.DecodingOptions(language=detected_language)
    result = whisper.decode(model, mel, options)

    return result.text, detected_language

def generate_response(text, language):
    system_message = "당신은 도움이 되는 AI 어시스턴트입니다. 사용자의 언어로 상세하고 포괄적인 응답을 제공해주세요."
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"다음 질문에 대해 {language}로 상세하고 포괄적으로 답변해주세요: {text}"}
        ],
        max_tokens=500,
        temperature=0.7,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6
    )
    return response.choices[0].message.content

def text_to_speech(text, language):
    speech_file_path = Path(__file__).parent / "response.mp3"
    
    voice = "alloy"  # 기본값
    if language == "ko":
        voice = "nova"  # 한국어
    elif language == "ja":
        voice = "onyx"  # 일본어
    elif language == "zh":
        voice = "alloy"  # 중국어 (alloy가 중국어를 지원)
    
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice=voice,
        input=text
    ) as response:
        response.stream_to_file(speech_file_path)
    
    return speech_file_path

def play_audio(audio_file):
    pygame.mixer.init()
    pygame.mixer.music.load(str(audio_file))
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.quit()

def main():
    device_index = select_microphone()
    
    while True:
        print("음성 입력을 시작하려면 Enter를 누르세요. (종료하려면 'q'를 입력하세요)")
        user_input = input()
        if user_input.lower() == 'q':
            break

        audio_file = record_audio(device_index=device_index)
        print("음성을 텍스트로 변환 중...")
        text, detected_language = transcribe_audio(audio_file)
        print(f"인식된 텍스트: {text}")

        print("응답 생성 중...")
        response = generate_response(text, detected_language)
        print(f"AI 응답: {response}")

        response_parts = [response[i:i+200] for i in range(0, len(response), 200)]
        for part in response_parts:
            print("음성 응답 생성 중...")
            response_audio = text_to_speech(part, detected_language)
            print("응답 재생 중...")
            play_audio(response_audio)

if __name__ == "__main__":
    main()
