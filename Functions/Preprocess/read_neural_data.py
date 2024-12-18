import os
import numpy as np
from pathlib import Path
import pickle

def read_groupdata(file_path, kword, ifsave = False):
    gfolders = [x for x in os.listdir(file_path) if kword in x and os.path.isdir(x)]
    if file_path[-1] not in ['/',"\\"]:
        file_path += '/'
    
    data = []
    tags = []
    for folder in gfolders:
        rdata = read_neural_data(os.path.join(file_path,folder))
        data.append(rdata['data'])
        tags.append(rdata['tags'])
    data = np.concatenate(data,axis=1)
    tags = np.concatenate(tags,axis=0)
    time = np.arange(data.shape[1]) / rdata['srate']

    rdata['data'] = data
    rdata['tags'] = tags
    rdata['time'] = time

    return rdata

def read_neural_data(file_path, ifsave = False):
    infoPath = os.path.join(file_path, 'Info.txt')
    DataPath_pre = [f for f in os.listdir(file_path) if f.startswith('Data')]
    DataPath = [os.path.join(file_path, name) for name in DataPath_pre]
    
    with open(infoPath, 'r') as fid:
        lines = fid.readlines()
        
    SampleRate = int(lines[1].split()[1])
    Preamp = int(lines[2].split()[1])
    Resolution = int(lines[3].split()[1])
    chCount = len(lines) - 10

    channelData = []
    digitalByte = []
    digitalBits = []

    for data_file in DataPath:
        with open(data_file, 'rb') as f:
            Data = np.fromfile(f, dtype=np.uint8)
            
        Data = Data[:len(Data) - len(Data) % (chCount * Resolution // 8 + 1)]
        part_channelData = np.zeros((len(Data) // (chCount * Resolution // 8 + 1), chCount))
        part_digitalByte = np.zeros((len(Data) // (chCount * Resolution // 8 + 1), 1), dtype=np.uint8)
        part_digitalBits = np.zeros((len(Data) // (chCount * Resolution // 8 + 1), 8), dtype=np.uint8)
        
        for j in range(0, chCount * Resolution // 8, Resolution // 8):
            if Resolution == 16:
                part_channelData[:, j // (Resolution // 8)] = (Data[j::chCount * Resolution // 8 + 1].astype(np.uint16) * 256 + Data[j + 1::chCount * Resolution // 8 + 1])
            elif Resolution == 24:
                part_channelData[:, j // (Resolution // 8)] = (Data[j::chCount * Resolution // 8 + 1].astype(np.uint32) * 65536 + Data[j + 1::chCount * Resolution // 8 + 1].astype(np.uint32) * 256 + Data[j + 2::chCount * Resolution // 8 + 1])

        
        part_digitalByte[:, 0] = Data[chCount * Resolution // 8::chCount * Resolution // 8 + 1]
        channelData.append(part_channelData)
        digitalByte.append(part_digitalByte)
    
    channelData = np.vstack(channelData)
    digitalByte = np.vstack(digitalByte)
    
    channelData[channelData > 2 ** (Resolution - 1)] -= 2 ** Resolution
    channelData = 8 * channelData / 2 ** Resolution
    
    time = np.arange(channelData.shape[0]) / SampleRate
    digitalBits = np.unpackbits(digitalByte, axis=1)
    
    rdata = {}
    rdata['data'] = channelData.T
    rdata['tags'] = np.squeeze(digitalByte)
    rdata['time'] = time
    rdata['srate'] = SampleRate
    rdata['nch'] = channelData.shape[1]

    if ifsave:
        Path(file_path).mkdir(parents=True, exist_ok=True)
        with open(file_path+'/data_raw.pkl', 'wb') as f:
            pickle.dump(rdata, f)

    return rdata