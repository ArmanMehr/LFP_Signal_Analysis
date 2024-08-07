import numpy as np
import scipy as sp
import pywt
import sys

sys.path.insert(0, '../Utils/')
import ts_utils

sig_ph = np.random.randn(1000)
sig_amp = np.random.randn(1000)
fph = [4,8]
famp = [20,80]
Fs = 2000

def PAC_varTime_ts(sig_ph,sig_amp,Fs,fph,famp):
    scales = np.arange(1, 128, 0.5)
    tfd_ph, f = pywt.cwt(sig_ph, scales = scales ,wavelet='morl', sampling_period=1/Fs)
    if np.all(sig_ph==sig_amp):
        tfd_amp = tfd_ph
    else:
        tfd_amp, f = pywt.cwt(sig_amp, scales = scales ,wavelet='morl', sampling_period=1/Fs)
    

def PAC_MI(Phase, Amp, nbins):
    
    Phase[Phase<0] = Phase[Phase<0] + 2*np.pi
    Lfamp = np.shape(Amp)[0]
    Lfph = np.shape(Phase)[0]
    PAC = np.zeros([Lfph,Lfamp])
    
    for fai in range(Lfamp):
        for fpi in range(Lfph):
            P = np.zeros(nbins)
            for i in range(nbins):
                bin_amp = Amp[fai,(Phase[fpi,:]>=(((i-1)/nbins)*2*np.pi)) & (Phase[fpi,:]<(i/nbins)*2*np.pi)]
                if not (bin_amp.size == 0):
                    P[i] = np.mean(bin_amp)
            P = P/np.sum(P)
            PAC[fpi,fai] = 1+np.sum(P[P!=0]*np.log2(P[P!=0]))/np.log2(nbins)
    
    return PAC