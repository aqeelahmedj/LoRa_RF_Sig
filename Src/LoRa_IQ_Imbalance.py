import numpy as np
import matplotlib.pyplot as plt
from numba import complex64


#simulation of the IQ gain imbalance for a LoRa symbol

# Parameters
N = 1024                  # Number of samples
fs = 1e6                # Sampling frequency
BW = 125e3                # LoRa bandwidth
f0 = BW / 2               # Tone frequency for test
t = np.arange(N) / fs

# LoRa-like upchirp baseband signal (simplified)
k = BW / N
phase = np.pi * k * t**2
s = np.exp(1j * phase)  # Ideal upchirp

# Introduce IQ imbalance
gain_mismatch_dB = 1.0     # gain mismatch in dB
phase_error_deg = 5.0      # phase error in degrees

gain_mismatch = 10**(gain_mismatch_dB/20)
phase_error = np.deg2rad(phase_error_deg)

# I/Q decomposition
I = np.real(s)
Q = np.imag(s)

# Apply gain and phase error
I_imb = I
Q_imb = gain_mismatch * (np.cos(phase_error) * Q + np.sin(phase_error) * I)

# Reconstruct imbalanced signal
s_imb = I_imb + 1j * Q_imb

# FFT
S = np.fft.fftshift(np.fft.fft(s))
S_imb = np.fft.fftshift(np.fft.fft(s_imb))
f = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))

# Power spectrum
PSD = 20*np.log10(np.abs(S) / np.max(np.abs(S)))
PSD_imb = 20*np.log10(np.abs(S_imb) / np.max(np.abs(S_imb)))

# Find desired and image tone powers
desired_idx = np.argmax(PSD_imb)
image_idx = N - desired_idx - 1  # Mirror index
P_desired = PSD_imb[desired_idx]
P_image = PSD_imb[image_idx]
IRR = P_desired - P_image

print(f"Measured IRR: {IRR:.2f} dB")


#using real LoRa IQ data

iq_data=np.fromfile(r"C:\Users\Umons\Documents\Github\Digital Signal Processing\Data\pluto_MKR2_26_6.bin", dtype=np.complex64)
print(len(iq_data))

few_frames =iq_data[17000000:20000000]

preambles=few_frames[621422:621422+8192]

np.save(r"C:\Users\Umons\Documents\Github\Digital Signal Processing\Data\one_pramble.npy", preambles)


plt.figure(figsize=(6,4))
plt.plot(preambles.real, label='I')
plt.plot(preambles.imag, label='Q')
plt.close()


#measuring IQ imbalance using IRR method

N = len(preambles)    #get lora iq samples
R = np.fft.fftshift(np.fft.fft(preambles))             #get FFT shifted
f = np.fft.fftshift(np.fft.fftfreq(N, 1/1e6))  # get the signal tone frequency values

PSD = 20*np.log10(np.abs(R)/np.max(np.abs(R)))   #need power of the recieved signal

desired_idx = np.argmax(PSD)
image_idx = N - desired_idx - 1  # mirror index

P_desired = PSD[desired_idx]
P_image = PSD[image_idx]

IRR = P_desired - P_image
print(f"Measured IRR: {IRR:.2f} dB")

#---This results in 14.95 dB... This shows there' some IQ imbalance in the LoRa data captured


#Theres' another way of estimating I and Q imbalance by taking gain and phase imblance
#this is from the paper--- Residual Channel Boosts Contrastive Learning for
#Radio Frequency Fingerprint Identification
#Rui Pan, Hui Chen, Guanxiong Shen, Member, IEEE, Hongyang Chen, Senior Member, IEEE


I = np.real(preambles)
Q = np.imag(preambles)

var_I = np.var(I)
var_Q = np.var(Q)

A_dB = 10 * np.log10(var_I / var_Q)
print(f"Estimated amplitude imbalance A: {A_dB:.2f} dB")



cov_IQ = np.mean(I * Q) - np.mean(I) * np.mean(Q)
rho_IQ = cov_IQ / np.sqrt(var_I * var_Q)

theta_rad = np.arcsin(rho_IQ)
P_deg = 2 * theta_rad * 180 / np.pi

print(f"Estimated phase imbalance P: {P_deg:.2f} degrees")


