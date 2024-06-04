import librosa
import noisereduce as nr
import soundfile as sf

def reduce_noise(audio_path):
    """Reduces noise in the audio file.

    Args:
        audio_path: The path to the input audio file.

    Returns:
        The path to the noise-reduced audio file.
    """
    # Load audio file
    audio_data, sr = librosa.load(audio_path)
    # Reduce noise
    reduced_noise = nr.reduce_noise(y=audio_data, sr=sr)
    # Save noise-reduced audio
    reduced_noise_path = "reduced_noise.wav"
    sf.write(reduced_noise_path, reduced_noise, sr)
    return reduced_noise_path

def resample_audio(audio_path, target_sr=16000):
    """Resamples the audio file to a target sample rate.

    Args:
        audio_path: The path to the input audio file.
        target_sr: The target sample rate (default is 16000 Hz).

    Returns:
        The path to the resampled audio file.
    """
    # Load audio file with original sample rate
    audio_data, sr = librosa.load(audio_path, sr=None)
    # Resample audio
    audio_data_resampled = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
    # Save resampled audio
    resampled_audio_path = "resampled_audio.wav"
    sf.write(resampled_audio_path, audio_data_resampled, target_sr)
    return resampled_audio_path

