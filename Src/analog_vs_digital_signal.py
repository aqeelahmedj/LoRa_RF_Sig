#this is the first practice program for my DSP course
#
import numpy as np
import matplotlib.pyplot as plt

#generating an example analog signal (continuous time)
t_cont = np.linspace(0, 1, 10000) #high resolution time
analog_signal = np.sin(2*np.pi*5*t_cont)
#digital signal (sampled and quantized)
fs=100   #sampling rate

t_disc = np.linspace(0,1, fs, endpoint=False)
digi_signal= np.sin(2*np.pi*5*t_disc)
n_level = 16
qunatized_sig = np.round(((digi_signal+1)/2)*(n_level-1))  #normalized tpo [0, n_level-1]
qunatized_sig = (qunatized_sig/(n_level-1))*2-1  # back to [-1, 1] range
#plotting
plt.figure(figsize=(5, 5))
#analog
plt.plot(t_cont, analog_signal, label='Analog Signal', color='blue')
plt.stem(t_disc, qunatized_sig, linefmt='r-', markerfmt='ro', basefmt='k-', label='Digital Signal')
plt.title('Analog Vs Digital Signal')
plt.xlabel('Time (s)')
plt.ylabel('Ampltidue')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

