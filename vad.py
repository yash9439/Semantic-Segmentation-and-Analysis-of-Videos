import argparse
import subprocess
import numpy as np
import scipy.io.wavfile
import scipy.ndimage
import webrtcvad

def perform_vad(audio_path, sample_rate=8000, mono=False, aggressiveness=3, window_size=0.02, gain=0.8, 
                 window_size_merge=1.0, window_size_max=0.2, energy_percentile=0.9):
    """
    Performs Voice Activity Detection (VAD) on an audio file using WebRTC VAD.

    Args:
        audio_path (str): Path to the audio file.
        sample_rate (int): Sample rate of the audio (in Hz).
        mono (bool): Whether to convert to mono audio.
        aggressiveness (int): Aggressiveness level for the VAD (0-3).
        window_size (float): VAD window size (in seconds).
        gain (float): Gain applied to the voice indicator track.
        window_size_merge (float): Window size for merging short voice segments.
        window_size_max (float): Maximum window size for filtering out noise.
        energy_percentile (float): Percentile used to determine energy threshold.

    Returns:
        None: Saves processed audio with voice indicator track to separate files.
    """

    num_channels = int(subprocess.check_output(['soxi', '-V0', '-c', audio_path])) if not mono else 1
    signal = np.frombuffer(subprocess.check_output(['sox', '-V0', audio_path, '-b', '16', '-e', 'signed', '--endian', 'little', '-r', str(sample_rate), '-c', str(num_channels), '-t', 'raw', '-']), 
                           dtype=np.int16).reshape(-1, num_channels)

    vad = webrtcvad.Vad(aggressiveness)

    percentile_window_size = 10.0
    frame_len = int(window_size * sample_rate)
    merge_filter_size = int(window_size_merge * sample_rate / frame_len)
    max_filter_size = int(window_size_max * sample_rate)
    percentile_filter_size = int(percentile_window_size * sample_rate)

    inflate = lambda voice, channel: np.repeat(voice, frame_len)[:len(channel)]

    for c, channel in enumerate(signal.T):
        voice = np.array([vad.is_speech(channel[sample_idx : sample_idx + frame_len].tobytes(), sample_rate) 
                          if sample_idx + frame_len <= len(signal) else False 
                          for sample_idx in range(0, len(channel), frame_len)])
        channel_abs = np.abs(channel)
        energy_threshold = np.quantile(channel_abs[inflate(voice, channel)], energy_percentile)

        voice &= (scipy.ndimage.filters.maximum_filter1d(channel_abs, max_filter_size, mode='constant') > energy_threshold)[::frame_len]
        voice = scipy.ndimage.morphology.binary_closing(voice, np.ones((merge_filter_size,), dtype=bool))

        output_path = audio_path + f'.{c}.wav'
        scipy.io.wavfile.write(output_path, sample_rate, 
                              np.vstack([channel, inflate(voice, channel).astype(channel.dtype) * int(channel.max() * gain)]).T)
        print(output_path)

