import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
import scipy.signal as signal


N1 = 16
N2 = 128
fs = 10**9
##fft equvalent 8192

f1 = fs/16*0.6
#f1 = 70*10**6
f2 = 230*10**6
f3 = 340*10**6
f4 = 480*10**6



sig_len = 2**15
t = np.arange(sig_len)*1.0/fs

sig = 1*np.sin(2*np.pi*f1*t)+1./4*np.sin(2*np.pi*f2*t)+1./16*np.sin(2*np.pi*f3*t)+1./128*np.sin(2*np.pi*f4*t)


#sig = signal.hilbert(sig)

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
data_pfb = np.zeros([N1, signal_taps.shape[0]-pfb_taps.shape[0]], dtype=complex)

for i in range(signal_taps.shape[0]-pfb_taps.shape[0]):
    data_pfb[:,i] = np.sum(pfb_taps*signal_taps[i:taps_len+i,:], axis=0)
   

data_out = ifft(data_pfb, axis=0)


spect_bands = 20*np.log10(fft(data_out[:,0:1024], axis=1))
spect_bands_aux = spect_bands.copy()
spect_bands[1:,0:512] = np.flip(spect_bands_aux[1:,:512], axis=1)
spect_bands[1:,512:] = spect_bands_aux[1:,512:][:,::-1]




df = fs/N1/10.**6
band = np.zeros([16, 1024])


def plot_band(ind):
    freq = np.linspace(df*(ind-0.5), df*(ind+.5), 1024, endpoint=False)
    if(ind%2==0):
        plt.plot(freq, spect_bands[ind,:])
    else:
        plt.plot(freq, spect_bands[ind,:])
    plt.show()


plt.figure()
for i in range(9):
    plot_band(i)
 
##Las bandas estan overlappeadas... k=0 va de 0, fs/(N1*2), 
##k1 = va de 0, fs/N1 y asi va avanzando
## para hacerlo facil toma bw=fs/N
## banda1 : (0, bw/2)
## banda2 : (0, bw)
## banda3 : (bw:2bw)  ...etc..

## a todo esto todas las bandas estan invertidas al paraecer
## eso es por usar la fft en vez de la ifft

##parece q esto tmb esta mal...
