'''
Design a low-pass FIR filter that removes high-frequency noise from an audio signal sampled at 8 kHz.
Specifications
Sampling Rate (fs): 8000 Hz

Cutoff Frequency (fc): 1000 Hz

Filter Type: FIR (Finite Impulse Response)

Filter Length: Choose an appropriate odd number (e.g., 21, 51, or 101 taps)

Design Method: Use window method (e.g., Hamming window)

Implementation Tool: Python + scipy.signal

Signal: Apply it to a noisy sine wave or short audio clip
asks
Generate a signal: clean sine wave at 500 Hz + added white noise.

Design the FIR filter using scipy.signal.firwin().

Apply the filter using scipy.signal.lfilter().

Plot time-domain signals before and after filtering.

Plot the frequency response of the filter using freqz

'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter, freqz

# Parameters
fs = 10000               # Sampling frequency in Hz
fc = 1000               # Cutoff frequency in Hz
numtaps = 51            # Filter length (number of taps)
duration = 1.0          # Signal duration in seconds
t = np.linspace(0, duration, int(fs*duration), endpoint=False)

# Generate a test signal: 500 Hz sine wave + white noise
f_signal = 500          # Signal frequency
signal_clean = np.sin(2 * np.pi * f_signal * t)
noise = np.random.normal(0, 0.5, signal_clean.shape)
signal_noisy = signal_clean + noise

# Design the FIR low-pass filter using Hamming window
fir_coeff = firwin(numtaps=numtaps, cutoff=fc, window='hamming', fs=fs)

# Apply the FIR filter to the noisy signal
signal_filtered = lfilter(fir_coeff, 1.0, signal_noisy)

# Plot time-domain signals
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(t, signal_clean)
plt.title('Clean Signal (500 Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 2)
plt.plot(t, signal_noisy)
plt.title('Noisy Signal (500 Hz + noise)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 3)
plt.plot(t, signal_filtered)
plt.title('Filtered Signal (Low-pass FIR)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

# Plot filter frequency response
w, h = freqz(fir_coeff, worN=8000, fs=fs)
plt.figure(figsize=(10, 4))
plt.plot(w, 20 * np.log10(np.abs(h)))
plt.title('Frequency Response of FIR Low-Pass Filter')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [dB]')
plt.grid()
plt.tight_layout()
plt.show()
