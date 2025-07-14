import os
import torch
from transformers import pipeline

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
chunk_length_s = int(os.environ.get('chunk_length_s', 30))

def model_fn(model_dir):
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model_dir,
        chunk_length_s=chunk_length_s,
        device=0 if torch.cuda.is_available() else -1
    )
    return asr_pipeline

def predict_fn(input_data, model):
    return model(input_data)
