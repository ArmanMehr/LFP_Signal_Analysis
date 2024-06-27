import pickle
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from copy import deepcopy
from pathlib import Path
from ..Basics import myfft

class preprocess_raw_data:
    def __init__(self, target_rate = 2000, filter_frange = [0.5,128], filter_order = 3, target_tags = [20,40], cleanLine_filtbw = 2, epoch_time_int = [-1,3], DataInfo = None, save_path = './'):
        self.target_rate = target_rate
        self.filter_frange = filter_frange
        self.filter_order = filter_order
        self.target_tags = target_tags
        self.cleanLine_filtbw = cleanLine_filtbw
        self.epoch_time_int = epoch_time_int
        self.save_path = save_path

    def fit(self, data):

        data_pp = deepcopy(data)

        print('Extract events based on target tags...')    
        data_pp = self.extract_all_events(data_pp)
        
        print('Downsample data to '+str(self.target_rate)+'Hz...')
        data_pp = self.downsample_rdata(data_pp)
        
        print('Band pass filter data based on filter_range:'+str(self.filter_frange)+'Hz...')
        data_pp = self.filter_rdata(data_pp, freq_range = self.filter_frange)
        
        print('Finding fft and psd of data...')
        
        myfft.getfft(data, ifPlot=True)
        while True:
            uans = input('Does it need notch filter at 50Hz?(y/n): ').lower()
            if uans == 'y':
                print('Applying notch filter on 50 with '+str(self.cleanLine_filtbw)+'Hz bandwidth...')
                data_pp = self.filter_rdata(data_pp, freq_range = [50 - self.cleanLine_filtbw/2, 50 + self.cleanLine_filtbw/2], ifBandStop = True)
                break
            elif uans == 'n':
                break
            else:
                print("The answer must be 'y' for yes or 'n' for no!")
        
        print('Epoch data based on target tags...')
        for ev in self.target_tags:
            data_pp['ev'+str(ev)] = {}
            epoched_data_temp = self.epoch_data(data_pp, target_tags = ev)
            data_pp['ev'+str(ev)]['data'] = epoched_data_temp['data']
            data_pp['ev'+str(ev)]['ntr'] = epoched_data_temp['ntr']

        data_pp['time'] = epoched_data_temp['time']
        del data_pp['data']

        # To be added (dataset info)        
        # if DataInfo:
        #     if not ''
        
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        with open(self.save_path+'/data_pp.pkl', 'wb') as f:
            pickle.dump(data_pp, f)

        print('The preprocessed data saved in "'+self.save_path+'/data_pp.pkl"')

        return data_pp

    def extract_all_events(self, data):
        
        data = deepcopy(data)
        
        if 'etimes' in data:
            return

        target_tags = np.unique(data['tags'])
        target_tags = np.delete(target_tags,np.where([target_tags==0])[0])
        event_idx = np.where(np.diff(data['tags'])>0)[0] + 1
        event_idx = event_idx[np.isin(data['tags'][event_idx],target_tags)]
        event_times = data['time'][event_idx]
        event_tags = data['tags'][event_idx]
        data['tags'] = {}
        data['tags']['etags'] = event_tags
        data['tags']['etimes'] = event_times
        return data

    def downsample_rdata(self, data):

        if self.target_rate == data['srate']:
            return data
        elif self.target_rate > data['srate']:
            raise ValueError('The new sampling rate is higher that the sampling rate of the origianl data')
        
        data = deepcopy(data)
        
        # Calculate Nyquist frequency for original and target sampling rates
        nyquist_original = 0.5 * data['srate']
        nyquist_target = 0.5 * self.target_rate

        # Design a low-pass filter
        cutoff_frequency = min(nyquist_original, nyquist_target)
        b, a = sp.signal.butter(self.filter_order, cutoff_frequency / nyquist_original)
        
        temp = sp.signal.resample_poly(data['data'][0,:], self.target_rate, data['srate'])
        resampled_signal = np.zeros([data['nch'], len(temp)])
        for ch in range(data['nch']):
            
            # Apply the low-pass filter
            filtered_signal = sp.signal.filtfilt(b, a, data['data'][ch,:])
            
            # Downsample the filtered signal
            resampled_signal[ch,:] = sp.signal.resample_poly(filtered_signal, self.target_rate, data['srate'])

        
        new_time = np.arange(0,resampled_signal.shape[1])/self.target_rate
        data['data'] = resampled_signal
        data['time'] = new_time
        data['srate'] = self.target_rate
        return data

    def filter_rdata(self, data, freq_range, ifBandStop = False):
        
        data = deepcopy(data)
        if not ifBandStop:
            if freq_range[-1] >= data['srate']/2:
                [num,den] = sp.signal.butter(self.filter_order, 2*np.array(freq_range[0])/data['srate'], btype='highpass')
            else:
                [num,den] = sp.signal.butter(self.filter_order, 2*np.array(freq_range)/data['srate'], btype='bandpass')
        else:
            [num,den] = sp.signal.butter(self.filter_order, 2*np.array(freq_range)/data['srate'], btype='bandstop')
        for ch in range(data['nch']):
            data['data'][ch,:] = sp.signal.filtfilt(num,den,data['data'][ch,:])
        return data

    def epoch_data(self, data, target_tags):
        
        data = deepcopy(data)
        data['tags']['etimes'] = data['tags']['etimes'][np.isin(data['tags']['etags'],target_tags)]
        data['tags']['etags'] = data['tags']['etags'][np.isin(data['tags']['etags'],target_tags)]
        tags_idx = np.int32(np.round(data['tags']['etimes']*data['srate']))
        n_int = np.int32(np.array([self.epoch_time_int[0],self.epoch_time_int[-1] - 1/data['srate']])*data['srate'])
        n_epoch = n_int[-1]-n_int[0]+1
        
        data_epoched = np.zeros([data['nch'],n_epoch,len(tags_idx)])
        for tr in range(len(tags_idx)):
            data_epoched[:,:,tr] = data['data'][:,(tags_idx[tr]+n_int[0]):(tags_idx[tr]+n_int[-1]+1)]
        
        data['data'] = data_epoched
        data['time'] = np.arange(self.epoch_time_int[0],self.epoch_time_int[-1],1/data['srate'])
        data['ntr'] = np.shape(data['data'])[2]

        data_epoched = (data_epoched - np.mean(data_epoched[:,data['time']<0,:], axis=1, keepdims=True))/np.std(data_epoched[:,data['time']<0,:], axis=1, keepdims=True)
        data['data'] = np.apply_along_axis(sp.stats.zscore , 1, data['data'])
        return data