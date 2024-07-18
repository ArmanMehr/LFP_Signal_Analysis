# DEV

import numpy as np
import scipy as sp

# TEST ZONE
twin = 0.5
tovp = 0.9


import pickle
with open('../../Data/data_pp.pkl','rb') as f:
    data_loaded = pickle.load(f)
Fs = data_loaded['srate']
win = np.ones(int(np.round(twin*Fs)))
novp = np.shape(window)[0]*tovp

sig1 = data_loaded['ev20']['data'][0,:,0]
sig2 = data_loaded['ev20']['data'][3,:,0]
# f,t,S = sp.signal.spectrogram(sig1,fs=Fs,window=win,noverlap=novp)
t = t+data_loaded['time'][0]
# plot = plt.pcolormesh(t,f,np.abs(S),shading='auto',clim = (0,0.2))
# plt.ylim([2,100])
# plt.colorbar(plot)

f,t,Sx = sp.signal.spectrogram(sig1,fs=Fs,window=win,noverlap=novp)
f,t,Sy = sp.signal.spectrogram(sig2,fs=Fs,window=win,noverlap=novp)
Sxy = Sx*np.conj(Sy)

for 
coh = np.mean(Sxy,1)/np.

def coherence_ts(sig1,sig2,Fs):
    f,t,Sx = sp.signal.spectrogram(sig1,fs=Fs,window=win,noverlap=novp)
    f,t,Sy = sp.signal.spectrogram(sig2,fs=Fs,window=win,noverlap=novp)
    
    for 
    Sxy = Sx*Sy



help(sp.signal.spectrogram)