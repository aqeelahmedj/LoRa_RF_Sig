import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, find_peaks
from toolz import sliding_window
from numpy.lib.stride_tricks import sliding_window_view


def lora_parameters(sf, bw, fs):
    """
    Compute LoRa parameters
    """
    symbol_duration = (2 ** sf) / bw
    symbol_len_samples = int(fs * symbol_duration)
    return symbol_len_samples, symbol_duration


def generate_chirp(length, fs, bw, up=True):
    """
    Generate an upchirp or downchirp
    """
    t = np.arange(length) / fs
    k = bw / ((2 ** sf) / bw)  # chirp rate: BW / Tsym
    phase = np.pi * k * t**2
    if up:
        chirp = np.exp(1j * phase)
    else:
        chirp = np.exp(-1j * phase)
    return chirp

def correlate_with_downchiprs(signal, downchirp, symbol_len):
    corr = np.zeros(len(signal)- symbol_len, dtype=np.float32)

    for i in range(len(corr)):
        segment = signal[i:i + symbol_len]

        c=np.abs(np.vdot(segment, downchirp))

        corr[i]=c

    return  corr


def find_preamble_start(corr, threshold_ratio=0.05):
    """
    Find first peak above threshold as preamble start
    """
    threshold = np.max(corr) * threshold_ratio
    peaks, _ = find_peaks(corr, height=threshold)
    if len(peaks) == 0:
        raise ValueError("No preamble detected.")
    return peaks[0]


def fine_align(signal, start_idx, upchirp, symbol_len):
    """
    Optionally refine alignment within one symbol by checking correlation around start_idx
    """
    window = 512  # samples left/right to check
    best_offset = 0
    best_value = 0
    for offset in range(-window, window):
        idx = start_idx + offset
        segment = signal[idx:idx + symbol_len]
        c = np.abs(np.vdot(segment, upchirp))
        if c > best_value:
            best_value = c
            best_offset = offset
    return best_offset


def lora_synchronize(iq_samples, sf, bw, fs, expected_symbols=10):
    """
    Full LoRa synchronization pipeline
    """
    # Parameters
    symbol_len, _ = lora_parameters(sf, bw, fs)
    # Chirps
    upchirp = generate_chirp(symbol_len, fs, bw, up=True)
    downchirp = np.conj(upchirp)
    # Coarse sync
    corr = correlate_with_downchiprs(iq_samples, downchirp, symbol_len)
    # Find preamble start
    coarse_start = find_preamble_start(corr)
    # Fine alignment
    fine_offset = fine_align(iq_samples, coarse_start, upchirp, symbol_len)
    preamble_start = coarse_start + fine_offset
    # Compute symbol boundaries
    symbol_starts = [
        preamble_start + n * symbol_len
        for n in range(expected_symbols)
    ]

    return preamble_start, symbol_starts




signals=np.fromfile(r"C:\Users\Umons\Documents\Github\Digital Signal Processing\Data\30_frames_HackRF_24_6.bin", dtype=np.complex64)
print(len(signals))

frame3=signals[4000000:60000000]

# Example usage
if __name__ == "__main__":
    fs = 1e6  # 1 MHz sampling rate
    bw = 125e3  # 125 kHz LoRa BW
    sf = 7  # Spreading factor

    preamble_start, symbol_starts = lora_synchronize(frame3, sf, bw, fs)

    print(f"Preamble start at sample: {preamble_start}")
    print(f"Symbol starts: {symbol_starts}")