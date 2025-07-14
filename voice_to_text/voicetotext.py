import whisper
import torch

def transcribe_audio(audio_path, model_size='base'):
    # Don't send the model to MPS to avoid sparse tensor error
    print("Loading model on CPU to avoid sparse tensor error on MPS...")

    model = whisper.load_model(model_size)  # keep on CPU
    result = model.transcribe(audio_path)

    print("\n--- Transcribed Text ---\n")
    print(result["text"])
    return result["text"]

if __name__ == "__main__":
    wav_file = "sound1.wav"  # replace with your WAV filename
    transcribe_audio(wav_file, model_size='base')
    wav_file2 = "LJ037-0171.wav"  # replace with your WAV filename
    transcribe_audio(wav_file2, model_size='base')

