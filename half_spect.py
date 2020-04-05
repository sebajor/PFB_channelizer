import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import scipy.signal as signal


N1 = 16
N2 = 128
fs = 10**9
##fft equvalent 8192

#f1 = 243835449.21875#*10**6 #20.5*10**6
f1 = 20*10**6
f2 = 220*10**6
f3 = 330*10**6
f4 = 480*10**6

sig_len = 2**15
t = np.arange(sig_len)*1.0/fs

sig = 1*np.sin(2*np.pi*f1*t)+1./4*np.sin(2*np.pi*f2*t)+1./16*np.sin(2*np.pi*f3*t)+1./128*np.sin(2*np.pi*f4*t)

##plot input signal

spect = fft(sig[0:16384])
freqs = np.linspace(0,fs/2,16384/2)
plt.figure()
plt.plot(freqs/10**6, 20*np.log10(np.abs(spect[0:8192])))
plt.title('signal spectrum')


#compute fir values

band = [0, 1.*fs/N1/2]
trans_width = 1.*fs/(N1*4*2)
n_taps = 512
edges = [band[0], band[1], band[1]+trans_width, 0.5*fs]
taps = signal.remez(n_taps, edges, [1, 0], Hz=fs)

w, h = signal.freqz(taps, [1], worN=2000)

##plot filter response
plt.figure()
plt.title('Filter response')
plt.plot(0.5*fs/10**6*w/np.pi, 20*np.log10(np.abs(h)))


### generate the taps of the pfb filter
pfb_taps = taps.reshape(n_taps/N1, N1)

#plot taps
"""
plt.figure()
plt.title('PFB taps response')
for i in range(N1):
    w, h = signal.freqz(pfb_taps[:,i], [1], worN=2000)
    plt.plot(0.5*fs/10**6*w/np.pi, 20*np.log10(np.abs(h)), label=('tap'+str(i)))
plt.legend()
    
"""


signal_taps = sig.reshape(sig_len/N1, N1)
taps_len = pfb_taps.shape[0]
data_pfb = np.zeros([N1, signal_taps.shape[0]-pfb_taps.shape[0]])

for i in range(signal_taps.shape[0]-pfb_taps.shape[0]):
    data_pfb[:,i] = np.sum(pfb_taps*signal_taps[i:taps_len+i,:], axis=0)
   
twidd = np.zeros([N1, N1], dtype=complex)
for i in range(N1):
    for j in range(N1):
        twidd[i,j] = np.exp(-1j*2*np.pi/N1*i*j)

    

data_out = np.zeros(data_pfb.shape, dtype=complex)

df = fs/N1/10.**6/2
freq = []
band = np.zeros([16, 512])
for i in range(N1):
    data_out[i,:] = np.dot(twidd[i,:],data_pfb)
    freq.append(np.linspace(df*i*2, df*(2*i+1), 512, endpoint=False))
    band[i,:] = 20*np.log10(np.abs(fft(data_out[i,0:1024])[:512]))




plt.show() 

def plot_band(ind):
    if(ind%2==0):
        plt.plot(freq[ind], band[ind,:])
    else:
        plt.plot(freq[ind], band[ind,:][::-1])
    plt.show()


plt.figure()
for i in range(8):
    plot_band(i)
 









