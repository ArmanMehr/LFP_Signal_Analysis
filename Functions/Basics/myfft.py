import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def myfft(data, event_tag = None, nfft = None, ifPlot = False):
    
    ppflag = 0
    if any(key.startswith('ev') for key in data):
        ppflag = 1
    
    if ppflag:
        if not event_tag:
            raise ValueError('event_tag input cannot be empty in preprocessed data.')
        
        data_f, freq = myfft_ts(data['ev'+str(event_tag)]['data'],data['srate'], ifPlot = ifPlot, nfft = nfft)
    else:
        data_f, freq = myfft_ts(data['data'], data['srate'], ifPlot = ifPlot, nfft = nfft)

    if ifPlot:
        return
    else:
        return data_f, freq

def myfft_ts(data_ts, Fs, nfft = None, ifPlot = False, channel_names = None):
    
    if not nfft:
        nfft = 8*Fs

    ndim = data_ts.ndim
    if ndim == 1:
        nch = 1
    elif ndim == 2:
        nch = data_ts.shape[0]
    elif ndim == 3:
        nch = data_ts.shape[0]
        ntr = data_ts.shape[2]
    else:
        ValueError("Input data must be a vector of time nsamples or a matrix with size of nchannels*nsamples or a tensor with size of nchannels*nsamples*ntrials")
    
    if not channel_names:
        channel_names = []
        for ch in range(nch):
            channel_names.append("Channel "+str(ch+1))

    if ndim < 3:
        Xf =  np.abs(sp.fft.fft(data_ts, nfft)/nfft)**2
    else:
        Xf = np.zeros([nch,nfft,ntr])
        for tr in range(ntr):
            Xf[:,:,tr] = np.abs(sp.fft.fft(data_ts[:,:,tr], nfft)/nfft)**2
        Xf = np.mean(Xf, axis = 2)
    
    if nch > 1:
        Xf = Xf[:,:nfft//2]
    else:
        Xf = Xf[:nfft//2]

    freq = sp.fft.fftfreq(nfft, 1/Fs)[:nfft//2]

    if ifPlot:
        if nch > 1:
            for ch in range(nch):
                plt.plot(freq,Xf[ch,:],label=channel_names[ch])
        else:
            plt.plot(freq,Xf,label=channel_names[ch])
        plt.legend()
        plt.xlim([0,64])
        plt.show()
    
    return Xf, freq