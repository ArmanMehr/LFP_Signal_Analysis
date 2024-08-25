import numpy as np
import scipy as sp
import pywt
from joblib import Parallel, delayed
from copy import deepcopy
from ..Utils import ts_utils
import time

def PAC_comodulogram_ts(sig_ph, sig_amp, Fs, fph = [3,8], famp = [20,80], nbins = 9, fres_param = 3.5, wavelet = 'cmor1.5-1.0'):

    sig_ph_n = sp.stats.zscore(sig_ph)
    sig_amp_n = sp.stats.zscore(sig_amp)
    
    k = 100
    scales = np.arange(1, k, fres_param)
    f = pywt.scale2frequency(wavelet = wavelet,scale=scales) * Fs
    while f[-1] > fph[0]:
        k += 10
        scales = np.arange(1, k, fres_param)
        f = pywt.scale2frequency(wavelet = wavelet,scale=scales) * Fs
        if (k > 2**14) or (fres_param <= 0):
            raise ValueError('Cannot find the best parameters for frequency resolution!')
    
    tfd_ph, f = pywt.cwt(sig_ph_n, scales = scales ,wavelet = wavelet, sampling_period=1/Fs)
    if np.all(sig_ph_n==sig_amp_n):
        tfd_amp = tfd_ph
    else:
        tfd_amp, f = pywt.cwt(sig_amp_n, scales = scales ,wavelet = 'morl', sampling_period=1/Fs)

    idx_ph = (f>=fph[0]) & (f<=fph[-1])
    idx_amp = (f>=famp[0]) & (f<=famp[-1])
    Phase = np.angle(tfd_ph[idx_ph,:])
    Amp = np.abs(tfd_amp[idx_amp,:])


    fph_vec = f[idx_ph]
    famp_vec = f[idx_amp]
    
    PAC = PAC_MI(Phase, Amp, nbins)

    return PAC, fph_vec, famp_vec

def PAC_varTime_ts(sig_ph, sig_amp, Fs, twin = 1, tovp = 0.5, fph = [3,8], famp = [20,80], nbins = 9, fres_param = 3.5, wavelet = 'cmor1.5-1.0'):

    nwin = int(np.round(twin*Fs))
    novp = int(np.round(twin*Fs*tovp))

    sig_ph_n = sp.stats.zscore(sig_ph)
    sig_amp_n = sp.stats.zscore(sig_amp)

    start = time.time()

    k = 100
    scales = np.arange(1, k, fres_param)
    f = pywt.scale2frequency(wavelet = wavelet, scale=scales) * Fs
    while f[-1] > fph[0]:
        k += 10
        scales = np.arange(1, k, fres_param)
        f = pywt.scale2frequency(wavelet = wavelet, scale=scales) * Fs
        if (k > 2**14) or (fres_param <= 0):
            raise ValueError('Cannot find the best parameters for frequency resolution!')
    
    tfd_ph, f = pywt.cwt(sig_ph_n, scales = scales, wavelet = wavelet, sampling_period=1/Fs)
    if np.all(sig_ph_n==sig_amp_n):
        tfd_amp = tfd_ph
    else:
        tfd_amp, f = pywt.cwt(sig_amp_n, scales = scales, wavelet = wavelet, sampling_period=1/Fs)
    
    idx_ph = (f>=fph[0]) & (f<=fph[-1])
    idx_amp = (f>=famp[0]) & (f<=famp[-1])
    Phase = np.angle(tfd_ph[idx_ph,:])
    Amp = np.abs(tfd_amp[idx_amp,:])

    fph_vec = f[idx_ph]
    famp_vec = f[idx_amp]
    Lfph = len(fph_vec)
    Lfamp = len(famp_vec)

    window_idx = ts_utils.window_it(np.arange(0,sig_ph.shape[0]), nwin, novp)
    Lt = window_idx.shape[0]
    t = np.arange(0,len(sig_ph))/Fs
    tout = t[np.int16(np.median(window_idx,axis=1))]*1e3

    end = time.time()
    runtime = end - start  # calculate the elapsed time
    print(f"Runtime: {runtime:.6f} seconds")

    PAC = np.zeros([Lt,Lfph,Lfamp])
    PAC = Parallel(n_jobs=8)(delayed(PAC_MI)(Phase[:,window_idx[ti,:]], Amp[:,window_idx[ti,:]], nbins) for ti in range(Lt))

    return PAC, tout, fph_vec, famp_vec

def PAC_MI(Phase, Amp, nbins):
    
    Phase_new = deepcopy(Phase)
    Phase_new[Phase_new<0] = Phase_new[Phase_new<0] + 2*np.pi
    Lfamp = np.shape(Amp)[0]
    Lfph = np.shape(Phase)[0]
    PAC = np.zeros([Lfph,Lfamp])
    
    for fai in range(Lfamp):
        for fpi in range(Lfph):
            P = np.zeros(nbins)
            for i in range(nbins):
                idx = np.where((Phase_new[fpi,:]>=((i/nbins)*2*np.pi)) & (Phase_new[fpi,:]<((i+1)/nbins)*2*np.pi))[0]
                if not (idx.shape[0] == 0):
                    P[i] = np.mean(Amp[fai,idx])
            P = P/np.sum(P)
            P = P[P!=0]
            PAC[fpi,fai] = 1+np.sum(P*np.log2(P))/np.log2(nbins)
    
    return PAC