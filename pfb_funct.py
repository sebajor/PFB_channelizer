import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
import scipy.signal as signal




def pfb(f):
    print(1.*f/10**6)
    N1 = 16
    N2 = 128
    fs = 10**9
    sig_len = 2**15
    t = np.arange(sig_len)*1.0/fs
    sig = 1*np.sin(2*np.pi*f*t)
    band = [0, 1.*fs/N1/2]
    trans_width = 1.*fs/(N1*4*2)
    n_taps = 512
    edges = [band[0], band[1], band[1]+trans_width, 0.5*fs]
    taps = signal.remez(n_taps, edges, [1, 0], Hz=fs)
    pfb_taps = taps.reshape(n_taps/N1, N1)
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
    for i in range(8):
        plot_band(i, spect_bands, df)
    
    plt.show()
    return spect_bands



def plot_band(ind, spect_bands, df):
    freq = np.linspace(df*(ind-0.5), df*(ind+0.5), 1024, endpoint=False)
    if(ind%2==0):
        plt.plot(freq, spect_bands[ind,:])
    else:
        plt.plot(freq, spect_bands[ind,:])



    




