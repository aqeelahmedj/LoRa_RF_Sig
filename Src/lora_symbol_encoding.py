import numpy as np
import matplotlib.pyplot as plt

# Parameters
BW = 125e3
SF = 7
M = 2**SF
Fs = BW
Tsym = M / BW
N = M
t = np.linspace(0, Tsym, N, endpoint=False)

payload_bits_len = 112  # e.g., 16 symbols * 7 bits per symbol
SNR_dB = -10

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

# Transmitter
payload_bits = np.random.randint(0, 2, payload_bits_len)
symbols = [int(''.join(map(str, payload_bits[i:i+SF])), 2) for i in range(0, len(payload_bits), SF)]

# Build TX signal
tx_signal = np.hstack([generate_chirp(s_offset=sym) for sym in symbols])

# AWGN
signal_power = np.mean(np.abs(tx_signal)**2)
noise_power = signal_power / (10**(SNR_dB/10))
noise = np.sqrt(noise_power/2) * (np.random.randn(len(tx_signal)) + 1j*np.random.randn(len(tx_signal)))
rx_signal = tx_signal + noise

# Receiver
detected_bits = []
for i in range(0, len(rx_signal), N):
    sym_chunk = rx_signal[i:i+N]
    dechirped = sym_chunk * ref_downchirp
    fft_out = np.fft.fft(dechirped)
    s_hat = np.argmax(np.abs(fft_out)) % M
    bits_hat = [int(b) for b in np.binary_repr(s_hat, SF)]
    detected_bits.extend(bits_hat)

# BER calculation
detected_bits = detected_bits[:payload_bits_len]
bit_errors = np.sum(payload_bits != detected_bits)
ber = bit_errors / payload_bits_len

print(f"Payload bits:      {payload_bits}")
print(f"Detected bits:     {detected_bits}")
print(f"Bit errors:        {bit_errors}")
print(f"Bit error rate:    {ber:.4f}")

# Optional: Plot TX and RX signals (real part)
plt.figure(figsize=(12,4))
plt.plot(np.real(tx_signal), label='TX signal (real)')
plt.plot(np.real(rx_signal), label='RX signal (real)', alpha=0.7)
plt.title("TX and RX Signal (Real Part)")
plt.legend()
plt.show()


