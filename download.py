import pytube
from moviepy.editor import VideoFileClip

def download_and_extract_audio(url):
    """Downloads a video from YouTube and extracts its audio.

    Args:
        url: The URL of the YouTube video.

    Returns:
        The path to the extracted audio file.
    """
    # Download video using pytube
    try:
        print("Downloading video...")
        yt = pytube.YouTube(url)
        video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        video.download(filename="video.mp4")
        print("Video downloaded successfully.")
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None

    # Extract audio using moviepy
    try:
        print("Extracting audio...")
        video_clip = VideoFileClip("video.mp4")
        audio_clip = video_clip.audio
        audio_clip.write_audiofile("audio.wav")
        print("Audio extracted successfully.")
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

    return "audio.wav"

