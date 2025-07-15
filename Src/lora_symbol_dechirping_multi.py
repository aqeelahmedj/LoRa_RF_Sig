import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
BW = 125e3
SF = 7
M = 2**SF
Fs = BW
Tsym = M / BW
N = M
t = np.linspace(0, Tsym, N, endpoint=False)

SNR_dB = 100
num_symbols = 20

# Helpers
def generate_chirp(s_offset=0, up=True):
    slope = BW / Tsym
    start_freq = -BW/2 + BW * s_offset / M
    if not up:
        slope = -slope
        start_freq = BW/2 - BW * s_offset / M
    inst_freq = start_freq + slope * t
    phase = 2 * np.pi * np.cumsum(inst_freq) / Fs
    return np.exp(1j * phase)

ref_downchirp = generate_chirp(s_offset=0, up=False)
time_ms = t * 1e6
freqs_kHz = np.linspace(0, BW/1e3, M)

# Prepare symbols
symbols = np.random.randint(0, M, num_symbols)
rx_signals = []
fft_magnitudes = []
detected_symbols = []
spectrograms = []

for s in symbols:
    tx_symbol = generate_chirp(s_offset=s)
    signal_power = np.mean(np.abs(tx_symbol)**2)
    noise_power = signal_power / (10**(SNR_dB/10))
    noise = np.sqrt(noise_power/2) * (np.random.randn(N) + 1j*np.random.randn(N))
    rx_symbol = tx_symbol + noise
    dechirped = rx_symbol * ref_downchirp
    fft_out = np.fft.fft(dechirped)
    fft_magnitude = np.abs(fft_out)
    detected_s = np.argmax(fft_magnitude) % M

    rx_signals.append(np.real(rx_symbol))
    fft_magnitudes.append(20*np.log10(fft_magnitude))
    detected_symbols.append(detected_s)
    # Spectrogram (use specgram return value correctly)
    Pxx, f, t_spec, im = plt.specgram(np.real(rx_symbol), NFFT=32, Fs=Fs, noverlap=16)
    spectrograms.append((f, t_spec, Pxx))
plt.clf()  # clear last specgram figure


# Animation function
def update(frame):
    ax1.clear()
    ax2.clear()
    ax3.clear()

    ax1.plot(time_ms, rx_signals[frame])
    ax1.set_title(f"Received Symbol {frame+1} (TX: S{symbols[frame]}) → Detected: S{detected_symbols[frame]}")
    ax1.set_xlabel("Time [ms]")
    ax1.set_ylabel("Amplitude")
    ax1.grid()

    ax2.plot(freqs_kHz, fft_magnitudes[frame])
    ax2.set_title("FFT of Dechirped Signal")
    ax2.set_xlabel("Frequency [kHz]")
    ax2.set_ylabel("Magnitude [dB]")
    ax2.grid()

    f, t_spec, Sxx = spectrograms[frame]
    ax3.pcolormesh(t_spec, f/1e3, 10*np.log10(Sxx), shading='gouraud')
    ax3.set_title("Spectrogram of Received Signal")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Frequency [kHz]")

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9))
ani = animation.FuncAnimation(fig, update, frames=num_symbols, interval=1000, repeat=False)
plt.tight_layout()
plt.show()
