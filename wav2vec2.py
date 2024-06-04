from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC
import torch
import librosa
import soundfile as sf

def wav2vec2(audio_path):
    """Retrieves word timestamps for a given audio file using Wav2Vec2.

    Args:
        audio_path: The path to the input audio file.

    Returns:
        A list of dictionaries with word and their start and end times.
    """
    # Import model, feature extractor, tokenizer
    model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

    # Load audio file
    audio_data, sr = librosa.load(audio_path, sr=16000)
    input_values = feature_extractor(audio_data, return_tensors="pt", sampling_rate=16000).input_values

    # Forward sample through model to get greedily predicted transcription ids
    logits = model(input_values).logits[0]
    pred_ids = torch.argmax(logits, axis=-1)

    # Retrieve word stamps (analogous commands for `output_char_offsets`)
    outputs = tokenizer.decode(pred_ids, output_word_offsets=True)

    # Compute `time_offset` in seconds as product of downsampling ratio and sampling_rate
    time_offset = model.config.inputs_to_logits_ratio / feature_extractor.sampling_rate

    word_offsets = [
        {
            "word": d["word"],
            "start_time": round(d["start_offset"] * time_offset, 2),
            "end_time": round(d["end_offset"] * time_offset, 2),
        }
        for d in outputs.word_offsets
    ]

    return word_offsets

