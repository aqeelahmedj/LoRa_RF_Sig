import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, find_peaks
from scipy import signal

iq_data=np.fromfile(r"C:\Users\Umons\Documents\Github\Digital Signal Processing\Data\pluto_MKR2_26_6.bin", dtype=np.complex64)
print(len(iq_data))





few_frames =iq_data[17000000:20000000]

# plt.figure(figsize=(6,4))
# plt.plot(few_frames.real, label='I')
# plt.plot(few_frames.imag, label='Q')
# plt.show()
#

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

def correlate_with_downchirp(signal, downchirp, symbol_len):
    """
    Slide over signal in symbol_len steps, correlate each segment with downchirp
    """
    corr = np.zeros(len(signal) - symbol_len, dtype=np.float32)
    for i in range(len(corr)):
        segment = signal[i:i + symbol_len]
        c = np.abs(np.vdot(segment, downchirp))
        corr[i] = c
    return corr


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
    corr = correlate_with_downchirp(iq_samples, downchirp, symbol_len)
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


# # Example usage
# if __name__ == "__main__":
#     fs = 1e6  # 1 MHz sampling rate
#     bw = 125e3  # 125 kHz LoRa BW
#     sf = 7  # Spreading factor
#
#     preamble_start, symbol_starts = lora_synchronize(few_frames, sf, bw, fs)
#
#     print(f"Preamble start at sample: {preamble_start}")
#     print(f"Symbol starts: {symbol_starts}")


preambles=few_frames[621422:621422+8192]

#time domain plot of the detected preamble
#it clearly show the perfect start of the preamble.. check the plot in Results folder
plt.figure(figsize=(6,4))
plt.plot(preambles.real, label='I')
plt.plot(preambles.imag, label='Q')
plt.savefig(r"C:\Users\Umons\Documents\Github\Digital Signal Processing\Results\peak_detected_preamble_start.png")
plt.close()

fs=1e6
# Compute the spectrogram for the selected frame
f, t, Zxx = signal.stft(preambles,
                 fs=fs,
                 window='boxcar',
                 nperseg=256,
                 noverlap=128,
                 nfft=256,
                 return_onesided=False)

f_shifted = np.fft.fftshift(f, axes=0)
spec = np.fft.fftshift(Zxx, axes=0)
spec_amp = np.log10(np.abs(spec) ** 2)

# Plot the spectrogram
plt.pcolormesh(t, f_shifted/1e6, spec_amp, shading='gouraud')
plt.title('STFT Magnitude')
plt.ylabel('Frequency [MHz]')
plt.xlabel('Time [sec]')
plt.colorbar(label='Magnitude (dB)')
plt.savefig(r"C:\Users\Umons\Documents\Github\Digital Signal Processing\Results\peak_detected_preamble_start_spectrogram.png")
plt.close()


