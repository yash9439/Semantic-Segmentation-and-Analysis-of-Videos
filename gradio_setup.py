import gradio as gr
import json
from download import *
from audioPreprocessing import *
from Wisper import *

def process_youtube_url(url):
    from download import download_and_extract_audio
    from audioPreprocessing import reduce_noise, resample_audio
    from Wisper import transcribe_audio_with_whisper

    # Step 1: Download and extract audio
    audio_file = download_and_extract_audio(url)
    if not audio_file:
        return "Failed to download and extract audio."

    # Step 2: Reduce noise in audio
    noise_reduced_audio = reduce_noise(audio_file)
    if not noise_reduced_audio:
        return "Failed to reduce noise in audio."

    # Step 3: Resample audio
    resampled_audio = resample_audio(noise_reduced_audio)
    if not resampled_audio:
        return "Failed to resample audio."

    # Step 4: Transcribe audio with timestamps
    transcriptions = transcribe_audio_with_whisper(resampled_audio, return_timestamps=True)
    if not transcriptions:
        return "Failed to transcribe audio."

    # Step 5: Format transcriptions
    sample_output_list = []
    chunk_id = 1
    for i in range(len(transcriptions)):
        start_time = transcriptions[i]['timestamp'][0]
        end_time = transcriptions[i]['timestamp'][1]
        chunk_length = round(end_time - start_time, 2)
        text = transcriptions[i]['text']
        sample_output_list.append({
            "chunk_id": chunk_id,
            "chunk_length": chunk_length,
            "text": text,
            "start_time": start_time,
            "end_time": end_time,
        })
        chunk_id += 1

    return sample_output_list

# Create Gradio interface
iface = gr.Interface(
    fn=process_youtube_url,
    inputs=gr.Textbox(lines=5, placeholder="Enter YouTube URL here..."),
    outputs=gr.JSON(),
    title="YouTube Audio Transcription",
    description="Enter a YouTube URL to transcribe its audio."
)

# Launch the interface
iface.launch()