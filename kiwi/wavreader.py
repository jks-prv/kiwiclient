# -*- python -*-

try:
    import collections.abc as collectionsAbc # python 3.6+
except ImportError:
    import collections as collectionsAbc
import struct
import numpy as np
try:
    from chunkmuncher import Chunk # python 3.13+
except ImportError:
    from chunk import Chunk

class KiwiIQWavError(Exception):
    pass

class KiwiIQWavReader(collectionsAbc.Iterator):
    def __init__(self, f):
        super(KiwiIQWavReader, self).__init__()
        self._frame_counter = 0
        self._last_gpssec   = -1
        try:
            self._f = open(f, 'rb')
            self._initfp(self._f)
        except:
            if self._f:
                self._f.close()
            raise

    def __del__(self):
        if self._f:
            self._f.close()

    def _initfp(self, file):
        self._file = Chunk(file, bigendian = 0)
        if self._file.getname() != b'RIFF':
            raise KiwiIQWavError('file does not start with RIFF id')
        if self._file.read(4) != b'WAVE':
            raise KiwiIQWavError('not a WAVE file')

        chunk = Chunk(self._file, bigendian = 0)
        if chunk.getname() != b'fmt ':
            raise KiwiIQWavError('fmt chunk is missing')

        self._proc_chunk_fmt(chunk)
        chunk.skip()

    ## for python3
    def __next__(self):
        return self.next()

    ## for python2
    def next(self):
        try:
            chunk = Chunk(self._file, bigendian = 0)
            if chunk.getname() != b'kiwi':
                raise KiwiIQWavError('missing KiwiSDR GNSS time stamp')

            self._proc_chunk_kiwi(chunk)
            chunk.skip()

            chunk = Chunk(self._file, bigendian = 0)
            if chunk.getname() != b'data':
                raise KiwiIQWavError('missing WAVE data chunk')

            tz = self._proc_chunk_data(chunk)
            chunk.skip()
            return tz
        except EOFError:
            raise StopIteration

    def process_iq_samples(self, t,z):
        ## print(len(t), len(z))
        pass

    def get_samplerate(self):
        return self._samplerate

    def _proc_chunk_fmt(self, chunk):
        wFormatTag, nchannels, self._samplerate, dwAvgBytesPerSec, wBlockAlign = struct.unpack('<HHLLH', chunk.read(chunk.getsize()-2))
        assert wFormatTag == 1 and nchannels == 2 and wBlockAlign == 4, 'this is not a KiwiSDR IQ wav file'

    def _proc_chunk_kiwi(self, chunk):
        self.last_gps_solution,dummy,gpssec,gpsnsec = struct.unpack('<BBII', chunk.read(10))
        self.gpssec = gpssec + 1e-9*gpsnsec

    def _proc_chunk_data(self, chunk):
        t = None
        z = np.frombuffer(chunk.read(chunk.getsize()), dtype=np.int16).astype(np.float32).view(np.complex64)/65535
        n = len(z)
        if self._last_gpssec >= 0:
            if self._frame_counter < 3:
                self._samplerate = n/(self.gpssec - self._last_gpssec)
            else:
                self._samplerate = 0.9*self._samplerate + 0.1*n/(self.gpssec - self._last_gpssec)

        if self._frame_counter >= 2:
            t = np.arange(start = self.gpssec,
                          stop  = self.gpssec + (n-0.5)/self._samplerate,
                          step  = 1/self._samplerate,
                          dtype = np.float64)
            ##t = self.gpssec + np.array(range(n))/self._samplerate
            self.process_iq_samples(t,z)

        self._last_gpssec = self.gpssec
        self._frame_counter += (self._frame_counter < 3)
        return t,z

def read_kiwi_iq_wav(filename):
    t = []
    z = []
    for _t,_z in KiwiIQWavReader(filename):
        if _t is None:
            continue
        t.append(_t)
        z.append(_z)
    return np.concatenate(t), np.concatenate(z)

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print('Usage: ...')

    [t,z]=read_kiwi_iq_wav(sys.argv[1])
    print (len(t),len(z), t[-1], z[-1], (t[-1]-t[-2])*1e6)
