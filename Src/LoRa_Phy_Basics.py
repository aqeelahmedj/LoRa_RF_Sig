import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

BW = 125e3
SF = 7
Fs = 8*BW
M  = 2**SF

print(f"M: {M}")   #numver chirps per symbol

Tsym = M / BW  #symbol duration

print(f"Tsym:{Tsym}")

N  = int (Tsym *Fs) #total samples in symbol

print(f"N: {N}")

t = np.linspace(0, Tsym, N, endpoint=False)

#let  us first plot unmodulated linear upchirp and downchirp

#instantaneous frequency

#−BW/2: start from the lowest frequency
# t / Tsym : Normalize the time vector
# t/Tsym *B: linear ramp .. the frequency sweep over entire BW linearly
# BW/2: reaches to maximum point

f_up =  -BW/2 + t /Tsym * BW + BW/2 #linear increase

f_down = BW / 2 - t/Tsym *  BW + BW/2 #linear decrease

phase1 = 2 * np.pi * np.cumsum(f_up) / Fs
phase2 = 2 * np.pi * np.cumsum(f_down) / Fs

upchirp = np.exp(1j*phase1)
downchirp = np.exp(1j*phase2)


plt.figure()
plt.subplot(2,1,1)
plt.plot(t*1e3, np.real(upchirp))
plt.title("A simple unmodulated LoRa Upchirp symbol ")
plt.xlabel("Time [ms]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()

plt.subplot(2,1,2)
plt.plot(t*1e3, np.real(downchirp))
plt.title("A simple unmodulated LoRa Downchirip symbol ")
plt.xlabel("Time [ms]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure()
plt.subplot(2,1,1)
plt.plot(t*1e3, f_up/1e3)
plt.title("Instantaneous Freq: unmodulated LoRa Upchirp symbol ")
plt.xlabel("Time [ms]")
plt.ylabel("Freq (kHz)")
plt.grid(True)
plt.tight_layout()

plt.subplot(2,1,2)
plt.plot(t*1e3, f_down/1e3)
plt.title("Instantaneous Freq: unmodulated LoRa Downchirip symbol ")
plt.xlabel("Time [ms]")
plt.ylabel("Freq (kHz)")
plt.grid(True)
plt.tight_layout()
plt.show()


s = 100
#starting frequency

f0 = -BW/2 + BW * s/M

#instantaneous frequency (wrap-around)
inst_freq = (f0 + BW * t / Tsym + BW/2) % BW - BW/2

phase = 2 *np.pi*np.cumsum(inst_freq) / Fs

css_symbol = np.exp(1j*phase)


plt.figure(figsize=(4, 8))
plt.subplot(2, 1, 1)
plt.plot(t*1e3, np.real(css_symbol))
plt.title(f"CSS Modulated Signal in Time Domain Symbol {s}")
plt.xlabel("Time [ms]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()


plt.subplot(2, 1, 2)
plt.plot(t*1e3, inst_freq / BW)
plt.title(f"Instantaneous Frequency of CSS Signal (Symbol {s})")
plt.xlabel("Time [ms]")
plt.ylabel("Frequency [Hz]")
plt.grid()
plt.tight_layout()
plt.show()


nfft = 32

f, tau, Sxx = spectrogram(css_symbol,
                          window='hamming',
                          fs=Fs,
                          nperseg=nfft,
                          noverlap=nfft//2,
                          return_onesided=False)
#f = np.fft.fftshift(f, axes=0)

Sxx_shifted = np.fft.fftshift(Sxx, axes=0)
f_shifted = np.fft.fftshift(f, axes=0)
Sxx_dB = 20 * np.log10(np.abs(Sxx_shifted) / np.max(np.abs(Sxx_shifted)))

plt.pcolormesh(tau*1e6, f_shifted/1e6, Sxx_dB, shading='gouraud', cmap='viridis')
plt.title(f"Spectrogram of CSS Modulated Signal (Symbol {s})")
plt.xlabel("Time [ms]")
plt.ylabel("Frequency [kHz]")
plt.colorbar(label="Power [dB]")
plt.grid()
plt.tight_layout()
plt.show()
