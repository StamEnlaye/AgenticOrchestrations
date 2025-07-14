from transformers import WhisperTokenizer, WhisperProcessor, AutoModelForSpeechSeq2Seq

model_name = "openai/whisper-base"
save_directory = "./whisper_model"

model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
tokenizer = WhisperTokenizer.from_pretrained(model_name)
processor = WhisperProcessor.from_pretrained(model_name)

model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
processor.save_pretrained(save_directory)
