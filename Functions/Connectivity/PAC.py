import numpy as np
import scipy as sp
import pywt
from tqdm import tqdm
from ..Utils import ts_utils

def PAC_varTime_ts(sig_ph, sig_amp, Fs, twin = 1, tovp = 0.8, fph = [3,8], famp = [20,80], nbins = 9, fres_param = 1, ifProgBar = True):

    nwin = np.round(twin*Fs)
    novp = np.round(twin*Fs*tovp)

    k = 100
    scales = np.arange(1, k, fres_param)
    f = pywt.scale2frequency(wavelet='morl',scale=scales) * Fs
    while (np.min(f) > fph[0]) or (np.shape(f[(f>=famp[0]) &(f<=famp[-1])])[0] < (famp[-1]-famp[0]+1)):
        k += 10
        scales = np.arange(1, k)
        f = pywt.scale2frequency(wavelet='morl',scale=scales) * Fs
        if (k > 2**14) or (fres_param <= 0):
            raise ValueError('Cannot find the best parameters for frequency resolution!')
    
    tfd_ph, f = pywt.cwt(sig_ph, scales = scales ,wavelet='morl', sampling_period=1/Fs)
    if np.all(sig_ph==sig_amp):
        tfd_amp = tfd_ph
    else:
        tfd_amp, f = pywt.cwt(sig_amp, scales = scales ,wavelet='morl', sampling_period=1/Fs)
    Phase = np.angle(tfd_ph[ (f>=fph[0]) & (f<=fph[-1]) ,:])
    Amp = np.abs(tfd_amp[ (f>=famp[0]) & (f<=famp[-1]) ,:])
    
    fph_vec = f[(f>=fph[0]) & (f<=fph[-1])]
    famp_vec = f[(f>=famp[0]) & (f<=famp[-1])]
    Lfph = len(fph_vec)
    Lfamp = len(famp_vec)


    window_idx = ts_utils.window_it(np.arange(0,sig_ph.shape[0]), nwin, novp)
    Lt = window_idx.shape[0]
    t = np.arange(0,len(sig_ph))/Fs
    tout = t[np.int16(np.median(window_idx,axis=1))]*1e3

    PAC = np.zeros([Lt,Lfph,Lfamp])

    if ifProgBar:
        pb = range(Lt)
        print('PAC calculation:')
        for ti in tqdm(pb):
            PAC[ti,:,:] = PAC_MI(Phase[:,window_idx[ti,:]], Amp[:,window_idx[ti,:]], nbins)
    else:
        for ti in range(Lt):
            PAC[ti,:,:] = PAC_MI(Phase[:,window_idx[ti,:]], Amp[:,window_idx[ti,:]], nbins)
    
    return PAC, tout, fph_vec, famp_vec

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