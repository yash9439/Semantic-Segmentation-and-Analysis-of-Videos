```markdown
# YouTube Audio Transcription & Segmentation

This project implements a pipeline for extracting semantically coherent audio-text pairs from YouTube videos. The pipeline leverages open-source tools and libraries to perform:

1. **Video Download & Audio Extraction:** Downloads the YouTube video and extracts its audio.
2. **Audio Preprocessing:** Reduces background noise and resamples the audio for compatibility with speech-to-text models.
3. **Transcription with Whisper:** Utilizes the Whisper model for accurate and robust transcription, including timestamping for each transcribed segment.
4. **Semantic Chunking:** Automatically segments the audio based on the timestamps provided by Whisper, ensuring each chunk contains a distinct word or phrase, preserving semantic coherence.

## Installation

```bash
pip install pytube noisereduce gradio pytube moviepy SpeechRecognition gentle nltk transformers pydub noisereduce gradio librosa matplotlib
```

## Usage (Use any 1)

1. **Run the Main Script:** Execute `main.ipynb` to process a YouTube video. 
   - Replace `url` in `main.ipynb` with the desired YouTube video URL.
   - The script downloads the video, extracts the audio, preprocesses it, transcribes the audio using Whisper, and outputs a JSON file containing the segmented audio-text pairs. 

2. **Launch the Gradio Interface:**  Run `gradio_setup.py` to launch a user-friendly web interface.
   - Enter a YouTube URL in the text box.
   - The interface will process the URL, transcribe the audio, and display the segmented audio-text pairs as JSON output.

## Example

**Input:** YouTube video URL: `https://www.youtube.com/watch?v=sHWsE1WnfyA`

**Output:**  A JSON file (`chunks.json`) containing the segmented audio-text pairs, for example:

```json
[
    {
        "chunk_id": 1,
        "chunk_length": 3.0,
        "text": " Shall I start off by wishing you a happy birthday in advance?",
        "start_time": 0.0,
        "end_time": 3.0
    },
    {
        "chunk_id": 2,
        "chunk_length": 1.0,
        "text": " Yes, sure.",
        "start_time": 3.0,
        "end_time": 4.0
    },
    ...
]
```

## Features

* **Automatic Segmentation:** Leverages Whisper's timestamping for efficient and semantically coherent chunking.
* **User-Friendly Interface:** Provides a Gradio web interface for easy interaction.
* **High Accuracy:** Utilizes the robust Whisper model for transcription.
* **Preprocessing:** Improves transcription quality through noise reduction and resampling.
* **Output Flexibility:**  Generates JSON output for further analysis or integration with other applications.

## Limitations

* **Monolingual Transcription:** Whisper is primarily designed for monolingual transcription. It struggles with audio containing multiple languages.
* **Error Propagation:** Errors in Whisper's transcription or timestamping can affect the accuracy of the segmentation.

## Future Development

* **Multilingual Support:** Explore methods for handling multilingual audio, potentially incorporating language detection and segmentation techniques or alternative models that handle multiple languages.
* **Advanced NLP Analysis:** Utilize more sophisticated NLP techniques for semantic analysis and chunk refinement.
* **Customizable Settings:** Implement user-configurable settings for chunk length, language selection, and model choice.
* **Interactive Visualization:**  Develop a visualization tool to display the segmented audio-text pairs, potentially including audio playback and timeline navigation.

