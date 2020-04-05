import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fft
import scipy.signal as signal

def plot_response(fs, w, h):
    "Utility function to plot response functions"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(0.5*fs*w/np.pi, 20*np.log10(np.abs(h)))
    ax.set_ylim(-40, 5)
    ax.set_xlim(0, 0.5*fs)
    ax.grid(True)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Gain (dB)')
    ax.set_title('Filter Respose')
    ax.axvline(fs/2/n_chann)

fs = 10**3

n_chann = 4

#low pass filter
cutoff = fs/2./n_chann
trans_width = 50
n_taps = 128

taps = signal.remez(n_taps, [0, cutoff, cutoff + trans_width, 0.5*fs], [1, 0], Hz=fs)

w, h = signal.freqz(taps, [1], worN=2000)

###input signal

t = np.arange(4096)*1.0/fs
"""
f1 = fs/2.*(1./4-0.125)
f2 = fs/2.*(1./2-0.125)
f3 = fs/2.*(3./4-0.125)
"""
f1 = 280
data = np.sin(2*np.pi*f1*t) #+ 1./16*np.sin(2*np.pi*f2*t) + 1./1024*np.sin(2*np.pi*f3*t)

fft_len = 1024
spect = fft.fft(data[0:fft_len])
freq = np.linspace(0, fs/2., fft_len, endpoint=False)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Input signal')
ax1.plot(20*np.log10(np.abs(spect[:fft_len/2])))




###pfb taps

pfb_tap = []
data_tap = []
for i in range(n_chann):
    pfb_tap.append(taps[i::n_chann])
    data_tap.append(data[i::n_chann])


##filter loop
tap0 = np.array([])
tap1 = np.array([])
tap2 = np.array([])
tap3 = np.array([])

out0 = []
out1 = []
out2 = []
out3 = []

out = []

size_tap = len(pfb_tap[0])

"""
##aca esta el queso....
for i in range(len(data_tap[0])):
    tap0 = np.insert(tap0,0, pfb_tap[0][i%size_tap]*data_tap[0][i])
    tap1 = np.insert(tap1,0, pfb_tap[1][i%size_tap]*data_tap[1][i])
    tap2 = np.insert(tap2,0, pfb_tap[2][i%size_tap]*data_tap[2][i])
    tap3 = np.insert(tap3,0, pfb_tap[3][i%size_tap]*data_tap[3][i])
    if(len(tap0)>size_tap):
        tap0 = tap0[:-1]
        tap1 = tap1[:-1]
        tap2 = tap2[:-1]
        tap3 = tap3[:-1]
        out.append(np.sum(tap0))
        out.append(np.sum(tap1))
        out.append(np.sum(tap2))
        out.append(np.sum(tap3))
        out0.append(np.sum(tap0))
        out1.append(np.sum(tap1))
        out2.append(np.sum(tap2))
        out3.append(np.sum(tap3))
""" 

for i in range(len(data_tap[0])):
    tap0 = np.insert(tap0,0, data_tap[0][i])
    tap1 = np.insert(tap1,0, data_tap[1][i])
    tap2 = np.insert(tap2,0, data_tap[2][i])
    tap3 = np.insert(tap3,0, data_tap[3][i])
    if(len(tap0)>size_tap):
        tap0 = tap0[:-1]
        tap1 = tap1[:-1]
        tap2 = tap2[:-1]
        tap3 = tap3[:-1]
        out.append(np.dot(tap0, pfb_tap[0]))
        out.append(np.dot(tap1, pfb_tap[1]))
        out.append(np.dot(tap2, pfb_tap[2]))
        out.append(np.dot(tap3, pfb_tap[3]))
    







out0 = np.array(out0)
out1 = np.array(out1)
out2 = np.array(out2)
out3 = np.array(out3)

out = np.array(out)

##fft
chan0 = []
chan1 = []
chan2 = []
chan3 = []

for i in range(len(out)/4):
    aux = fft.fft(out[4*i:4*(i+1)])
    chan0.append(aux[0])
    chan1.append(aux[1])
    chan2.append(aux[2])
    chan3.append(aux[3])



chan0 = np.array(chan0)
chan1 = np.array(chan1)
chan2 = np.array(chan2)
chan3 = np.array(chan3)

###plot spectrum of the bands


fig1 = plt.figure()
ax1 = fig1.add_subplot(141)
ax2 = fig1.add_subplot(142)
ax3 = fig1.add_subplot(143)
ax4 = fig1.add_subplot(144)

ax1.set_title('chan1')
ax2.set_title('chan2')
ax3.set_title('chan3')
ax4.set_title('chan4')

spect1 = fft.fft(chan0)
spect2 = fft.fft(chan1)
spect3 = fft.fft(chan2)
spect4 = fft.fft(chan3)


ax1.plot(20*np.log10(np.abs(spect1[:len(spect1)/2])+1))
ax2.plot(20*np.log10(np.abs(spect2[:len(spect2)/2])+1))
ax3.plot(20*np.log10(np.abs(spect3[:len(spect3)/2])+1))
ax4.plot(20*np.log10(np.abs(spect4[:len(spect4)/2])+1))



        


#plot_response(fs, w, h)
plt.show()
