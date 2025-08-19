#!/usr/bin/env python3

import soundfile,struct,sys
import numpy as np

## convert kiwirecorder IQ wav files to .c2 file for wsprd

def decode_kiwi_wav_filename(fn):
    """ input:  kiwirecorder wav file name
        output: .c2 filename (length 14) and frequency in kHz"""
    fields = fn.split('/')[-1].split('_')
    date_and_time = fields[0]
    c2_filename = date_and_time[2:8] + '_' + date_and_time[9:13] + '.c2'
    freq_kHz = int(fields[1])/1e3
    return c2_filename,freq_kHz

def convert_kiwi_wav_to_c2(fn):
    """ input:  kiwirecorder IQ wav file name with sample rate 375 Hz
        output: .c2 file for wsprd"""
    c2_filename,freq_kHz = decode_kiwi_wav_filename(fn)
    iq,fs = soundfile.read(fn)
    assert(fs == 375) ## sample rate has to be 375 Hz
    assert(iq.shape[1] == 2) ## works only for IQ mode recordings
    iq[:,1] *= -1
    iq.resize((45000,2), refcheck=False)
    f = open(c2_filename, 'wb')
    f.write(struct.pack('<14sid', c2_filename.encode(), 2, freq_kHz))
    f.write(iq.astype(np.float32).flatten().tobytes())
    f.close()
    print('in=' + fn + ' out='+c2_filename)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('convert kiwirecorder IQ wav files to .c2 file for wsprd')
        print('USAGE: ' + sys.argv[0]+ ' .wav filenames')
    for fn in sys.argv[1:]:
        convert_kiwi_wav_to_c2(fn)

