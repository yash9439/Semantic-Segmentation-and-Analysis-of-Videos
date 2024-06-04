from transformers import pipeline
import torch
from audioPreprocessing import resample_audio
import librosa

def transcribe_audio_with_whisper(audio_path, return_timestamps=False, language="en"):
    """Transcribes audio using Whisper.

    Args:
        audio_path: The path to the input audio file.
        return_timestamps: Boolean flag to indicate whether to return timestamps.
        language: The language of the input audio.

    Returns:
        The transcriptions with or without timestamps.
    """
    # Load and resample audio file
    resampled_audio_path = resample_audio(audio_path, target_sr=16000)
    audio_data, sr = librosa.load(resampled_audio_path, sr=16000)

    # Load and configure the pipeline
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-medium",
        chunk_length_s=30,  # Process audio in 30-second chunks
        device=device
    )

    # Process audio in chunks and get transcriptions
    if return_timestamps:
        transcriptions_with_timestamps = pipe(audio_data, batch_size=8, return_timestamps=True, generate_kwargs = {"task":"transcribe", "language":f"<|{language}|>"})["chunks"]
        return transcriptions_with_timestamps
    else:
        transcriptions = pipe(audio_data, batch_size=8, generate_kwargs = {"task":"transcribe", "language":f"<|{language}|>"})["text"]
        return transcriptions


