import numpy as np
import matplotlib.pyplot as plt

# Parameters
BW = 125e3              # LoRa bandwidth (Hz)
SF = 7                  # Spreading Factor
M = 2**SF               # Number of symbols
Fs = BW                 # Sampling rate (Hz)
Tsym = M / BW           # Symbol duration (s)
N = M                   # FFT size, same as number of symbols
t = np.linspace(0, Tsym, N, endpoint=False)

s = 10               # Transmitted symbol (0 <= s < M)
SNR_dB = 10         # Signal-to-Noise Ratio in dB

# Helper: generate upchirp or downchirp
def generate_chirp(t, BW, Tsym, Fs, s_offset=0, up=True):
    slope = BW / Tsym
    start_freq = -BW/2 + BW * s_offset / M
    if not up:
        slope = -slope
        start_freq = BW/2 - BW * s_offset / M
    inst_freq = start_freq + slope * t
    phase = 2 * np.pi * np.cumsum(inst_freq) / Fs
    chirp = np.exp(1j * phase)
    return chirp

# Generate transmitted upchirp with symbol offset
tx_upchirp = generate_chirp(t, BW, Tsym, Fs, s_offset=s, up=True)

# Add AWGN
signal_power = np.mean(np.abs(tx_upchirp)**2)
noise_power = signal_power / (10**(SNR_dB/10))
noise = np.sqrt(noise_power/2) * (np.random.randn(N) + 1j*np.random.randn(N))
rx_signal = tx_upchirp + noise

# Generate reference downchirp
ref_downchirp = generate_chirp(t, BW, Tsym, Fs, up=False)

# Dechirp (correct)
dechirped_down = rx_signal * ref_downchirp

fft_out = np.fft.fft(dechirped_down)
fft_magnitude = np.abs(fft_out)

freqs_hz = np.fft.fftfreq(N, d=1/Fs)[:M]
fft_magnitude = fft_magnitude[:M]
freqs_hz = np.linspace(0, BW, M)

# Estimate symbol
estimated_bin = np.argmax(fft_magnitude)
estimated_symbol = estimated_bin % M
detected_freq = estimated_bin * BW / M

plt.figure()
plt.subplot(4, 1,1)

plt.plot(t*1e3, np.real(tx_upchirp))
plt.title(f"Transmitted CSS LoRa Symbol:'{s}' at SNR: {SNR_dB}")
plt.xlabel("Time [ms]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()

plt.subplot(4, 1,2)
plt.plot(t*1e3, np.real(ref_downchirp))
plt.title(f"Dechirping with a reference downchirp")
plt.xlabel("Time [ms]")
plt.ylabel("Amplitude")
plt.grid(True)


plt.subplot(4, 1,3)
plt.plot(t*1e3, np.real(rx_signal))
plt.title(f"Received LoRa Symbol: {s}")
plt.xlabel("Frequency [kHz]")
plt.ylabel("Magnitude [dB]")
plt.grid()
plt.tight_layout()


plt.subplot(4, 1,4)
plt.plot(freqs_hz/1e3, 20*np.log10(fft_magnitude))
plt.title(f"FFT of Dechirped Signal Estimated Symbol Peak: {estimated_symbol}")
plt.xlabel("Frequency [kHz]")
plt.ylabel("Magnitude [dB]")
plt.grid()
plt.show()

print(f"Transmitted symbol: {s}")
print(f"Estimated symbol:   {estimated_symbol}")
print(f"Estimated frequency: {detected_freq/1e3:.2f} kHz")
