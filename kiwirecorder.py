#!/usr/bin/env python3
## -*- python -*-

##
## FIXME:
## netcat and camping should support the usual suspects:
##      IQ-swap, endian-reversal, squelch, option to include GPS data ...
##

VERSION = 'v1.8'

import array, logging, os, struct, sys, time, copy, threading, os
import gc
import math
import numpy as np
from copy import copy
from traceback import print_exc
import png
from kiwi import KiwiSDRStream, KiwiWorker
import optparse as optparse
from optparse import OptionParser
from optparse import OptionGroup
from queue import Queue,Empty
import json

try:
    import urllib.parse as urllib
except ImportError:
    import urllib

HAS_PyYAML = True
try:
    ## needed for the --agc-yaml and --scan-yaml options
    import yaml
    if yaml.__version__.split('.')[0] < '5':
        print('wrong PyYAML version: %s < 5; PyYAML is only needed when using the --agc-yaml or --scan-yaml option' % yaml.__version__)
        raise ImportError
except ImportError:
    ## (only) when needed an exception is raised, see below
    HAS_PyYAML = False

HAS_RESAMPLER = True
try:
    ## if available use libsamplerate for resampling
    from samplerate import Resampler
    ## NB: MemoryError, because on some systems get:
    ## "MemoryError: Cannot allocate write+execute memory for ffi.callback().
    ## You might be running on a system that prevents this.
    ## For more information, see https://cffi.readthedocs.io/en/latest/using.html#callbacks"
except (ImportError, MemoryError):
    ## otherwise linear interpolation is used
    HAS_RESAMPLER = False

try:
    if os.environ['USE_LIBSAMPLERATE'] == 'False':
        HAS_RESAMPLER = False
except KeyError:
    pass

def clamp(x, xmin, xmax):
    if x < xmin:
        x = xmin
    if x > xmax:
        x = xmax
    return x

def by_dBm(e):
    return e['dBm']

def _init_common(self, options, type):
    self._options = options
    self._type = type
    freq = options.frequency
    #logging.info("%s:%s freq=%d" % (options.server_host, options.server_port, freq))
    self._freq = freq
    self._freq_offset = options.freq_offset
    self._start_ts = None
    self._start_time = None
    self._options.writer_init = False

    self._squelch = Squelch(options) if options.sq_thresh is not None or options.scan_yaml is not None else None
    self._scan_continuous = False
    self._options.scan_state = 'INIT'
    if options.scan_yaml is not None:
        if 'threshold' in options.scan_yaml:
            threshold = options.scan_yaml['threshold']
        else:
            self._scan_continuous = True
            threshold = 0
        self._squelch = [Squelch(options).set_threshold(threshold) for _ in range(len(options.scan_yaml['frequencies']))]
        if 'pbc' in options.scan_yaml:
            self._options.freq_pbc = options.scan_yaml['pbc']
            logging.info('YAML file: --pbc')

    self._last_gps = dict(zip(['last_gps_solution', 'dummy', 'gpssec', 'gpsnsec'], [0,0,0,0]))
    self._resampler = None
    self._kiwi_samplerate = False
    self._gnss_performance = GNSSPerformance()
    self._camp_chan = options.camp_chan

def _setup_rx_snd_params(self, user):
    if self._options.no_api:
        self._setup_no_api()
        return
    
    self.set_name(self._options.user)

    self.set_freq(self._freq)

    if self._options.agc_gain != None: ## fixed gain (no AGC)
        self.set_agc(on=False, gain=self._options.agc_gain)
    elif self._options.agc_yaml_file != None: ## custon AGC parameters from YAML file
        self.set_agc(**self._options.agc_yaml)
    else: ## default is AGC ON (with default parameters)
        self.set_agc(on=True)

    if self._options.compression is False:
        self._set_snd_comp(False)

    if self._options.nb is True or self._options.nb_test is True:
        gate = self._options.nb_gate
        if gate < 100 or gate > 5000:
            gate = 100
        nb_thresh = self._options.nb_thresh
        if nb_thresh < 0 or nb_thresh > 100:
            nb_thresh = 50
        self.set_noise_blanker(gate, nb_thresh)

    if self._options.de_emp is True:
        self.set_de_emp(1)

    self._output_sample_rate = self._sample_rate

    if self._squelch:
        if type(self._squelch) == list: ## scan mode
            for s in self._squelch:
                s.set_sample_rate(self._sample_rate, not self._compression or self._stereo)
        else:
            self._squelch.set_sample_rate(self._sample_rate, not self._compression or self._stereo)

    if self._options.test_mode:
        self._set_stats()

    if self._options.resample > 0 and not HAS_RESAMPLER:
        _setup_resampler(self)

    if self._options.devel is not None:
        for pair in self._options.devel.split(','):
            vals = pair.split(':')
            if len(vals) != 2:
                raise Exception("--devel arg \"%s\" needs to be format \"[0-7]:float_value\"" % pair)
            which = int(vals[0])
            value = float(vals[1])
            if not (0 <= which <= 7):
                raise Exception("--devel first arg \"%d\" of \"[0-7]:float_value\" is out of range" % which)
            self._send_message('SET devl.p%d=%.9g' % (which, value))

def _setup_resampler(self):
    if self._options.resample > 0:
        if not HAS_RESAMPLER:
            self._output_sample_rate = self._options.resample
            self._ratio = float(self._output_sample_rate)/self._sample_rate
            logging.warning("CAUTION: libsamplerate not available; low-quality linear interpolation will be used for resampling.")
            logging.warning("See the README file instructions to build the Kiwi samplerate module.")
            logging.warning('resampling from %g to %d Hz (ratio=%f)' % (self._sample_rate, self._options.resample, self._ratio))
        else:
            if hasattr(self._resampler, 'kiwi_samplerate'):
                self._kiwi_samplerate = True
            if self._kiwi_samplerate is True:
                logging.warning("Using Kiwi high-quality samplerate module.")
                self._ratio = self._options.resample / self._sample_rate
            else:
                ## work around a bug in python-samplerate:
                ##  the following makes sure that ratio * 512 is an integer
                ##  at the expense of resampling frequency precision for some resampling frequencies (it's ok for WSPR 375 Hz)
                fs = 10*round(self._sample_rate/10) ## rounded sample rate
                ratio = self._options.resample / fs
                n = 512 ## KiwiSDR block length for samples
                m = round(ratio*n)
                self._ratio = m/n
                logging.warning('CAUTION: using python-samplerate instead of Kiwi samplerate module containing fixes.')
            self._output_sample_rate = self._ratio * self._sample_rate
            logging.warning('resampling from %g to %g Hz (ratio=%f)' % (self._sample_rate, self._output_sample_rate, self._ratio))

def _write_wav_header(self, fp, filesize, samplerate, num_channels, is_kiwi_wav):
    # always 2-channels if camping because can't predict what camped channel will do (mono vs stereo)
    nchans = 2 if self._camping else num_channels
    samplerate = int(samplerate+0.5)
    fp.write(struct.pack('<4sI4s', b'RIFF', filesize - 8, b'WAVE'))
    bits_per_sample = 16
    byte_rate       = samplerate * nchans * bits_per_sample // 8
    block_align     = nchans * bits_per_sample // 8
    fp.write(struct.pack('<4sIHHIIHH', b'fmt ', 16, 1, nchans, samplerate, byte_rate, block_align, bits_per_sample))
    if not is_kiwi_wav:
        fp.write(struct.pack('<4sI', b'data', filesize - 12 - 8 - 16 - 8))

def enqueue_input(inp, q):
    for line in iter(inp.readline, b''):
        q.put(line)
    inp.close()

class RingBuffer(object):
    def __init__(self, len):
        self._array = np.zeros(len, dtype='float64')
        self._index = 0
        self._is_filled = False

    def insert(self, sample):
        self._array[self._index] = sample
        self._index += 1
        if self._index == len(self._array):
            self._is_filled = True
            self._index = 0

    def is_filled(self):
        return self._is_filled

    def applyFn(self, fn):
        return fn(self._array)

    def max_abs(self):
        return np.max(np.abs(self._array))

class GNSSPerformance(object):
    def __init__(self):
        self._last_solution = -1
        self._last_ts = -1
        self._num_frames = 0
        self._buffer_dt_per_frame = RingBuffer(10)
        self._buffer_num_frames   = RingBuffer(10)

    def analyze(self, filename, gps):
        ## gps = {'last_gps_solution': 1, 'dummy': 0, 'gpsnsec': 886417795, 'gpssec': 466823}
        self._num_frames += 1
        if gps['last_gps_solution'] == 0 and self._last_solution != 0:
            ts = gps['gpssec'] + 1e-9 * gps['gpsnsec']
            msg_gnss_drift = ''
            dt = 0
            if self._last_ts != -1:
                dt = ts - self._last_ts
                if dt < -12*3600*7:
                    dt += 24*3600*7
                if abs(dt) < 10:
                    self._buffer_dt_per_frame.insert(dt / self._num_frames)
                    self._buffer_num_frames.insert(self._num_frames)
                if self._buffer_dt_per_frame.is_filled():
                    std_dt_per_frame  = self._buffer_dt_per_frame.applyFn(np.std)
                    mean_num_frames   = self._buffer_num_frames.applyFn(np.mean)
                    msg_gnss_drift = 'std(clk drift)= %5.1f m' % (3e8 * std_dt_per_frame * mean_num_frames)

            logging.info('%s: (%2d,%3d) t_gnss= %16.9f dt= %16.9f %s'
                         % (filename, self._last_solution, self._num_frames, ts, dt, msg_gnss_drift))
            self._num_frames = 0
            self._last_ts    = ts

        self._last_solution = gps['last_gps_solution']


class Squelch(object):
    def __init__(self, options):
        self._options  = options
        self._status_msg  = not options.quiet
        self._threshold   = options.sq_thresh
        self._squelch_tail = options.squelch_tail ## in seconds
        self._ring_buffer = RingBuffer(65)
        self._squelch_on_seq = None
        self.set_sample_rate(12000.0, False) ## default setting

    def set_threshold(self, threshold):
        self._threshold = threshold
        return self

    def set_sample_rate(self, fs, not_comp_or_stereo):
        bsize = 512 if not_comp_or_stereo else 2048
        self._tail_delay  = round(self._squelch_tail*fs/bsize) ## seconds to number of buffers

    def process(self, seq, rssi):
        if not self._ring_buffer.is_filled() or self._squelch_on_seq is None:
            self._ring_buffer.insert(rssi)
            if self._ring_buffer.is_filled() and self._options.scan_yaml_file and self._options.scan_state == 'INIT':
                self._options.scan_state = 'WAIT'
        median_nf   = self._ring_buffer.applyFn(np.median)
        rssi_thresh = median_nf + self._threshold
        is_open     = self._squelch_on_seq is not None
        if is_open:
            rssi_thresh -= 6
        rssi_green = rssi >= rssi_thresh
        if rssi_green:
            self._squelch_on_seq = seq
            is_open = True
        if self._status_msg:
            if self._options.log_level != 'debug':
                sys.stdout.write('\r  Median: %6.1f Thr: %6.1f %s ' % (median_nf, rssi_thresh, ("s", "S")[is_open]))
            else:
                t = time.strftime('%H:%M:%S', time.gmtime())
                sys.stdout.write('Median:    %6.1f Thr: %6.1f %s %s\n' % (median_nf, rssi_thresh, t, ("s", "S")[is_open]))
            sys.stdout.flush()
            self._need_nl = True
        if not is_open:
            return False
        if seq > self._squelch_on_seq + self._tail_delay:
            logging.info("Squelched")
            self._squelch_on_seq = None
            return False
        return is_open

## -------------------------------------------------------------------------------------------------

class KiwiSoundRecorder(KiwiSDRStream):
    def __init__(self, options):
        super(KiwiSoundRecorder, self).__init__()
        _init_common(self, options, 'SND')

    def _setup_rx_params(self):
        _setup_rx_snd_params(self, self._options.user)

    def _squelch_status(self, seq, rssi, fmt):
        if not self._options.quiet:
            fmt = "" if fmt is None or self._options.log_level != 'debug' else fmt
            if True or self._options.log_level != 'debug':
                sys.stdout.write('\rBlock: %08x, RSSI: %6.1f %s ' % (seq, rssi, fmt))
            else:
                t = time.strftime('%H:%M:%S', time.gmtime())
                sys.stdout.write('Block: %08x, RSSI: %6.1f %s %s\n' % (seq, rssi, t, fmt))
            sys.stdout.flush()
            self._need_nl = True

        is_open = True
        if self._squelch:
            if type(self._squelch) == list: ## scan mode
                self._options.scan_list = True
                if self._options.scan_state == 'INIT':
                    self._squelch[self._options.scan_index].process(seq, rssi)
                    is_open = False
                if self._options.scan_state == 'WAIT':
                    is_open = False
                    now = time.time()
                    if now - self._options.scan_time > self._options.scan_yaml['wait']:
                        self._options.scan_time = now
                        self._options.scan_state = 'DWELL'
                if self._options.scan_state == 'DWELL':
                    if self._scan_continuous:
                        is_open = True
                        check_time = True   # scanning never stops
                    else:
                        is_open = self._squelch[self._options.scan_index].process(seq, rssi)
                        check_time = not is_open    # stops scanning when squelch open
                    now = time.time()
                    if check_time and now - self._options.scan_time > self._options.scan_yaml['dwell']:
                        self._options.scan_index = (self._options.scan_index + 1) % len(self._options.scan_yaml['frequencies'])
                        self.set_freq(self._options.scan_yaml['frequencies'][self._options.scan_index])
                        self._options.scan_time = now
                        self._options.scan_state = 'WAIT'
                        self._start_ts = None
                        self._start_time = None
            else: ## single channel mode
                self._options.scan_list = False
                is_open = self._squelch.process(seq, rssi)
                if not is_open:
                    self._start_ts = None
                    self._start_time = None

        if self._squelch and type(self._squelch) == list: ## scan mode
            t = time.strftime('%H:%M:%S', time.gmtime())
            first = '\r' if self._options.quiet else ''
            last = '\n' if self._options.log_level == 'debug' else ''
            sys.stdout.write("%sScan: %s %s %-5s %g kHz      %s" % (first, t, ("s", "S")[is_open], self._options.scan_state, self._freq, last))
            self._need_nl = True
            sys.stdout.flush()
        return is_open


    def _process_audio_samples(self, seq, samples, rssi, fmt):
        is_open = self._squelch_status(seq, rssi, fmt)
        if not is_open:
            return

        if self._options.resample > 0:
            if HAS_RESAMPLER:
                ## libsamplerate resampling
                if self._resampler is None:
                    self._resampler = Resampler(converter_type='sinc_best')
                    _setup_resampler(self)
                samples = np.round(self._resampler.process(samples, ratio=self._ratio)).astype(np.int16)
            else:
                ## resampling by linear interpolation
                n  = len(samples)
                xa = np.arange(round(n*self._ratio))/self._ratio
                xp = np.arange(n)
                samples = np.round(np.interp(xa,xp,samples)).astype(np.int16)

        self._write_samples(samples, {})

    def _process_iq_samples(self, seq, samples, rssi, gps, fmt):
        if not self._squelch_status(seq, rssi, fmt):
            return

        if gps != None:
            ##print gps['gpsnsec']-self._last_gps['gpsnsec']
            self._last_gps = gps

        ## convert list of complex numbers into an array
        s = np.zeros(2*len(samples), dtype=np.int16)
        s[0::2] = np.real(samples).astype(np.int16)
        s[1::2] = np.imag(samples).astype(np.int16)

        if self._options.resample > 0:
            if HAS_RESAMPLER:
                ## libsamplerate resampling
                if self._resampler is None:
                    self._resampler = Resampler(channels=2, converter_type='sinc_best')
                    _setup_resampler(self)
                s = self._resampler.process(s.reshape(len(samples),2), ratio=self._ratio)
                s = np.round(s.flatten()).astype(np.int16)
            else:
                ## resampling by linear interpolation
                n  = len(samples)
                m  = int(round(n*self._ratio))
                xa = np.arange(m)/self._ratio
                xp = np.arange(n)
                s  = np.zeros(2*m, dtype=np.int16)
                s[0::2] = np.round(np.interp(xa,xp,np.real(samples))).astype(np.int16)
                s[1::2] = np.round(np.interp(xa,xp,np.imag(samples))).astype(np.int16)

        self._write_samples(s, gps)

        if gps != None:
            # no GPS or no recent GPS solution
            last = gps['last_gps_solution']
            if last == 255 or last == 254:
                self._options.status = 3

    def _update_wav_header(self):
        with open(self._get_output_filename(), 'r+b') as fp:
            fp.seek(0, os.SEEK_END)
            filesize = fp.tell()
            fp.seek(0, os.SEEK_SET)

            # fp.tell() sometimes returns zero. _write_wav_header writes filesize - 8
            if filesize >= 8:
                _write_wav_header(self, fp, filesize, self._output_sample_rate, self._num_channels, self._options.is_kiwi_wav)

    def _write_samples(self, samples, *args):
        """Output to a file on the disk."""
        if self._options.test_mode or self._options.scan_state == 'WAIT':
            return
        now = time.gmtime()
        sec_of_day = lambda x: 3600*x.tm_hour + 60*x.tm_min + x.tm_sec
        dt_reached = self._options.dt != 0 and self._start_ts is not None and sec_of_day(now)//self._options.dt != sec_of_day(self._start_ts)//self._options.dt
        if self._start_ts is None or (self._options.filename == '' and dt_reached):
            #logging.info("dt_reached=%d,%d start_ts=%d" % (dt_reached, self._options.dt, self._start_ts == None))
            self._start_ts = now
            self._start_time = time.time()
            # Write a static WAV header
            with open(self._get_output_filename(), 'wb') as fp:
                _write_wav_header(self, fp, 100, self._output_sample_rate, self._num_channels, self._options.is_kiwi_wav)
            if self._options.is_kiwi_tdoa:
                # NB for TDoA support: MUST be a print to stdout (i.e. not a logging.info to stderr)
                print("file=%d %s" % (self._options.idx, self._get_output_filename()))
            else:
                logging.info("Started a new file: %s" % self._get_output_filename())
        with open(self._get_output_filename(), 'ab') as fp:
            if self._options.is_kiwi_wav:
                gps = args[0]
                if gps != None:
                    self._gnss_performance.analyze(self._get_output_filename(), gps)
                    fp.write(struct.pack('<4sIBBII', b'kiwi', 10, gps['last_gps_solution'], 0, gps['gpssec'], gps['gpsnsec']))
                    sample_size = samples.itemsize * len(samples)
                    fp.write(struct.pack('<4sI', b'data', sample_size))
            # TODO: something better than that
            samples.tofile(fp)
        self._update_wav_header()

    def _on_gnss_position(self, pos):
        pos_record = False
        if self._options.dir is not None:
            pos_dir = self._options.dir
            pos_record = True
        else:
            if os.path.isdir('gnss_pos'):
                pos_dir = 'gnss_pos'
                pos_record = True
        if pos_record:
            station = 'kiwi_noname' if self._options.station is None else self._options.station
            pos_filename = pos_dir +'/'+ station + '.txt'
            with open(pos_filename, 'w') as f:
                station = station.replace('-', '_')   # since Octave var name
                f.write("d.%s = struct('coord', [%f,%f], 'host', '%s', 'port', %d);\n"
                        % (station,
                           pos[0], pos[1],
                           self._options.server_host,
                           self._options.server_port))

## -------------------------------------------------------------------------------------------------

# CAUTION: must match order in kiwi rx/mode.h
DX_MODE = {
    'am':0, 'amn':1, 'usb':2, 'lsb':3, 'cw':4, 'cwn':5, 'nbfm':6, 'iq':7, 'drm':8,
    'usn':9, 'lsn':10, 'sam':11, 'sau':12, 'sal':13, 'sas':14, 'qam':15, 'nnfm':16, 'amw':17
}

DX_7DAYS   = 0x0000fe00
DX_MODE_16 = 0x00040000

class KiwiWaterfallRecorder(KiwiSDRStream):
    def __init__(self, options):
        super(KiwiWaterfallRecorder, self).__init__()
        self._options = options
        self._type = 'W/F'
        freq = options.frequency
        #logging.info "%s:%s freq=%d" % (options.server_host, options.server_port, freq)
        self._freq = freq
        self._freq_offset = options.freq_offset
        self._start_ts = time.gmtime()
        self._start_time = None
        self._last_gps = dict(zip(['last_gps_solution', 'dummy', 'gpssec', 'gpsnsec'], [0,0,0,0]))
        self._wf_pass = 0
        self._p_unsorted_avg = []
        self._rows = []
        self._cmap_r = array.array('B')
        self._cmap_g = array.array('B')
        self._cmap_b = array.array('B')

        # Kiwi color map
        for i in range(256):
            if i < 32:
                r = 0
                g = 0
                b = i*255/31
            elif i < 72:
                r = 0
                g = (i-32)*255/39
                b = 255
            elif i < 96:
                r = 0
                g = 255
                b = 255-(i-72)*255/23
            elif i < 116:
                r = (i-96)*255/19
                g = 255
                b = 0
            elif i < 184:
                r = 255
                g = 255-(i-116)*255/67
                b = 0
            else:
                r = 255
                g = 0
                b = (i-184)*128/70

            self._cmap_r.append(clamp(int(round(r)), 0, 255))
            self._cmap_g.append(clamp(int(round(g)), 0, 255))
            self._cmap_b.append(clamp(int(round(b)), 0, 255))

    def _setup_rx_params(self):
        self.set_freq(self._freq)
        baseband_freq = self._remove_freq_offset(self._freq)
        self._set_zoom_cf(self._options.zoom, baseband_freq)
        self._set_maxdb_mindb(-10, -110)    # needed, but values don't matter
        self._set_wf_speed(self._options.speed)
        if self._options.no_api:
            self._setup_no_api()
            return
        self._set_wf_comp(self._options.wf_comp)
        self._set_wf_interp(self._options.interp)
        self.set_name(self._options.user)

        if self._options.dx_list != None:
            min, max = self._options.dx_list
            self._send_message('SET MARKER db=0 min=%f max=%f zoom=0 width=1024 eibi_types_mask=0x0 filter_tod=0 anti_clutter=0' % (float(min), float(max)))
        if self._options.dx_add:
            # 65202 0xfeb2 fe=7-days b=t11(fax) 2=usb
            logging.debug('mode=%s' % self.get_mod())
            mode = DX_MODE[self.get_mod()]
            if mode >= 16:
                mode = mode - 16
                mode = mode | DX_MODE_16
            flags = DX_7DAYS | ((self._options.dx_type & 0xf) << 4) | mode
            self._send_message('SET DX_UPD g=-1 f=%.2f lo=0 hi=0 o=0 s=0 fl=%d b=0 e=2400 i=%sx n=%sx p=%sx'
                % (self.get_frequency(), flags, urllib.quote(self._options.dx_ident), urllib.quote(self._options.dx_notes), urllib.quote(self._options.dx_params)))
        if self._options.dx_del != None:
            self._send_message('SET DX_UPD g=%d f=-1' % self._options.dx_del)
        if self._options.dx and not self._options.dx_list:
            time.sleep(1)
            self.exit()
        
        self._start_time = time.time()
        span = self.zoom_to_span(self._options.zoom)
        start = baseband_freq - span/2
        stop  = baseband_freq + span/2
        if self._options.wf_cal is None:
            self._options.wf_cal = -13      # pre v1.550 compatibility
        logging.info("wf samples: start|center|stop %.1f|%.1f|%.1f kHz, zoom %d, span %d kHz, rbw %.3f kHz, cal %d dB"
              % (start, baseband_freq, stop, self._options.zoom, span, span/self.WF_BINS, self._options.wf_cal))
        if start < 0 or stop > self.MAX_FREQ:
            s = "Frequency and zoom values result in span outside 0 - %d kHz range" % (self.MAX_FREQ)
            raise Exception(s)
        if self._options.wf_png is True:
            logging.info("--wf_png: mindb %d, maxdb %d, cal %d dB" % (self._options.mindb, self._options.maxdb, self._options.wf_cal))

    def _process_mkr(self, value):
        dx_filename = "dx.json"
        logging.info("writing file %s..." % dx_filename)
        try:
            #print(value)
            m = json.loads(value)
        except Exception as e:
            print(e)
            print("json.loads FAILED for:")
            print(value)
            return
        #print(json.dumps(m, indent=1))
        entries = 0
        with open(dx_filename, 'w') as fp:
            for e in m:
                if 'tc' in e:   # skip header entry
                    continue
                freq = e['f']
                if (freq) < 0:
                    print(e)
                flags = e['fl']
                mode = \
                    [ "AM", "AMN", "USB", "LSB", "CW", "CWN", "NBFM", "IQ", "DRM", "USN", "LSN", "SAM", "SAU", "SAL", "SAS", "QAM", "NNFM" ] \
                    [flags & 0xf | (0x10 if (flags & 0x40000) else 0)]
                type = (flags & 0x1f0) >> 4
                ident = urllib.unquote(e['i']) if 'i' in e else ''
                name = urllib.unquote(e['n']) if 'n' in e else ''

                lo = e['lo'] if 'lo' in e else 0
                hi = e['hi'] if 'hi' in e else 0
                lo_s = hi_s = ''
                if lo != 0 or hi != 0:
                    lo_s = ', "lo":%d' % lo
                    hi_s = ', "hi":%d' % hi
                offset = ', "o":%d' % e['o'] if 'o' in e and e['o'] != 0 else ''
                dow = (flags & 0xfe00) >> 9
                dow_s = ', "d0":%d' % dow if dow != 0 and dow != 0x7f else ''
                b0 = e['b'] if 'b' in e else 0
                e0 = e['e'] if 'e' in e else 0
                b_s = e_s = ''
                if b0 != 0 or (e0 != 0 and e0 != 2400):
                    b_s = ', "b0":%d' % b0
                    e_s = ', "e0":%d' % e0
                ext = ', "p":"%s"' % urllib.unquote(e['p']) if 'p' in e and e['p'] != '' else ''

                #print('[%.2f, "%s", "%s", "%s", {"T%d":1}%s%s%s%s%s%s%s],' % (freq, mode, ident, name, type, lo_s, hi_s, offset, dow_s, b_s, e_s, ext))
                s = '[%.2f, "%s", "%s", "%s", {"T%d":1}%s%s%s%s%s%s%s]' % (freq, mode, ident, name, type, lo_s, hi_s, offset, dow_s, b_s, e_s, ext)
                print('--dx-del=%d => %s' % (e['g'], s))
                fp.write('%s,\n' % s)
                entries += 1
        logging.info("wrote %d entries" % entries)
        self.exit()

    def _waterfall_color_index_max_min(self, db_value):
        db_value = clamp(db_value + self._options.wf_cal, self._options.mindb, self._options.maxdb)
        relative_value = db_value - self._options.mindb
        fullscale = self._options.maxdb - self._options.mindb
        fullscale = fullscale if fullscale != 0 else 1      # can't be zero
        value_percent = relative_value / fullscale
        return clamp(int(round(value_percent * 255)), 0, 255)
    
    def _process_waterfall_samples(self, seq, samples):
        baseband_freq = self._remove_freq_offset(self._freq)
        span = self.zoom_to_span(self._options.zoom)
        start = baseband_freq - span/2
        stop  = baseband_freq + span/2
        nbins = len(samples)
        bins = nbins-1
        i = 0
        pwr = []
        p_unsorted = []
        pixels = array.array('B')
        do_wf = self._options.wf_png and (not self._options.wf_auto or (self._options.wf_auto and self._wf_pass != 0))

        for s in samples:
            dBm = int(s) - 255      # 55..255 => -200..0 dBm
            #print(f'{i}: {s} {type(s)} {dBm} {type(dBm)} ')
            if i > 2 and dBm > -190:    # skip DC offset notch in first two bins and also masked areas
                pwr.append({ 'dBm':dBm, 'i':i })
            p_unsorted.append(dBm)
            i = i+1
            
            if do_wf:
                ci = self._waterfall_color_index_max_min(dBm)
                pixels.append(self._cmap_r[ci])
                pixels.append(self._cmap_g[ci])
                pixels.append(self._cmap_b[ci])
        
        if self._options.snr != 0:
            pwr.sort(key = by_dBm)
            length = len(pwr)
            noise = pwr[int(0.50 * length)]['dBm']
            signal = pwr[int(0.95 * length)]['dBm']
            logging.info("--snr: pass %d SNR %d" % (self._wf_pass, signal - noise))

            if self._wf_pass == 0:
                self._p_unsorted_avg = p_unsorted
                self._wf_pass = self._wf_pass+1
                return
            
            if self._wf_pass < self._options.snr:
                #print(f'pass={self._wf_pass} len={len(self._p_unsorted_avg)}')
                i = 0
                for _p in self._p_unsorted_avg:
                    self._p_unsorted_avg[i] = self._p_unsorted_avg[i] + p_unsorted[i]
                    i = i+1
                self._wf_pass = self._wf_pass+1
                return

            run = 0
            self.prev_dBm = -190
            th = -82
            rg = 25
            runlen = 10
            fs = start
            kpp = span/1024
            have_run = 0

            i = 0
            for _p in self._p_unsorted_avg:
                self._p_unsorted_avg[i] = self._p_unsorted_avg[i] / self._options.snr
                i = i+1

            i = 0
            for dBm in self._p_unsorted_avg:
                dBm = dBm + self._options.wf_cal
                p = self.prev_dBm
                f = fs + i*kpp
                diff = abs(dBm-p)
                #logging.debug(f'{f:.0f}: {dBm:.0f} {p:.0f} {diff:.0f}')
                if dBm >= th and p >= th:
                    if diff <= rg:
                        if run == 0:
                            self._last_start = f
                            self._last_idx = i
                        run = run+1
                        #logging.debug(f'  {run}')
                    else:
                        if run > runlen:
                            have_run = run
                        run = 0
                        #logging.debug(f'  RESET delta')
                else:
                    #if run:
                        #logging.debug(f'  RESET level')
                    if run > runlen:
                        have_run = run
                    run = 0
                if have_run:
                    #logging.info(f'--snr: run={have_run} {self._last_start:.0f}-{f:.0f}({have_run*kpp:.0f}) ==================================================')
                    for j in range(self._last_idx, i+1):
                        self._p_unsorted_avg[j] = -190      # notch the run
                    have_run = 0
                self.prev_dBm = dBm
                i = i+1

            pwr = []
            for i in range(1024):
                dBm = self._p_unsorted_avg[i]
                if i > 2 and dBm > -190:    # skip DC offset notch in first two bins and also masked & notched areas
                    pwr.append({ 'dBm':dBm, 'i':i })

        pwr.sort(key = by_dBm)
        length = len(pwr)
        #print(f'{pwr}')
        pmin = pwr[0]['dBm'] + self._options.wf_cal
        bmin = pwr[0]['i']
        pmax = pwr[length-1]['dBm'] + self._options.wf_cal
        bmax = pwr[length-1]['i']
        span = self.zoom_to_span(self._options.zoom)
        start = baseband_freq - span/2
        
        if (not self._options.wf_png and not self._options.quiet) or (self._options.wf_png and self._options.not_quiet):
            noise = pwr[int(0.50 * length)]['dBm']
            signal = pwr[int(0.95 * length)]['dBm']
            logging.info("wf samples: %d bins, min %d dB @ %.2f kHz, max %d dB @ %.2f kHz, SNR %d"
                  % (nbins, pmin, start + span*bmin/bins, pmax, start + span*bmax/bins, signal - noise))

        if self._options.snr != 0:
            self._stop = True
            return

        if self._options.wf_peaks > 0:
            with open(self._get_output_filename("_peaks.txt"), 'a') as fp:
                for i in range(self._options.wf_peaks):
                    j = length-1-i
                    bin_i = pwr[j]['i']
                    bin_f = float(bin_i)/bins
                    fp.write("%d %.2f %d  " % (bin_i, start + span*bin_f, pwr[j]['dBm'] + self._options.wf_cal))
                fp.write("\n")

        if self._options.wf_png and self._options.wf_auto and self._wf_pass == 0:
            noise = pwr[int(0.50 * length)]['dBm']
            signal = pwr[int(0.95 * length)]['dBm']
            # empirical adjustments
            signal = signal + 30
            if signal < -80:
                 signal = -80
            noise -= 10
            self._options.mindb = noise
            self._options.maxdb = signal
            logging.info("--wf_auto: mindb %d, maxdb %d, cal %d dB" % (self._options.mindb, self._options.maxdb, self._options.wf_cal))
        self._wf_pass = self._wf_pass+1
        if do_wf is True:
            self._rows.append(pixels)

    def _close_func(self):
        if self._options.wf_png is True:
            self._flush_rows()
        if self._options.wf_peaks > 0:
            logging.info("--wf-peaks: writing to file %s" % self._get_output_filename("_peaks.txt"))

    def _flush_rows(self):
        if not self._rows:
            return
        filename = self._get_output_filename(".png")
        logging.info("--wf_png: writing file %s" % filename)
        while True:
            with open(filename, 'wb') as fp:
                try:
                    png.Writer(len(self._rows[0]) // 3, len(self._rows)).write(fp, self._rows)
                    break
                except KeyboardInterrupt:
                    pass

## -------------------------------------------------------------------------------------------------

class KiwiExtensionRecorder(KiwiSDRStream):
    def __init__(self, options):
        super(KiwiExtensionRecorder, self).__init__()
        self._options = options
        self._type = 'EXT'
        self._freq = None
        self._start_ts = None
        self._start_time = time.time()

    def _setup_rx_params(self):
        self.set_name(self._options.user)
        # rx_chan deprecated, sent for backward compatibility only
        self._send_message('SET ext_switch_to_client=%s first_time=1 rx_chan=0' % self._options.extension)
        if self._options.no_api:
            self._setup_no_api()
            return

        if (self._options.extension == 'DRM'):
            if self._kiwi_version is not None and self._kiwi_version >= 1.550:
                self._send_message('SET lock_set')
                self._send_message('SET monitor=0')
                self._send_message('SET send_iq=0')
                self._send_message('SET run=1')
            else:
                raise Exception("KiwiSDR server v1.550 or later required for DRM")

        if self._options.ext_test:
            self._send_message('SET test=1')

    def _process_ext_msg(self, log, name, value):
        prefix = "EXT %s = " % name if name != None else ""
        if log is True:
            logging.info("recv %s%s" % (prefix, value))
        else:
            sys.stdout.write("%s%s%s\n" % ("\n" if self._need_nl else "", prefix, value))
            self._need_nl = False if self._need_nl is True else False

    def _process_ext(self, name, value):
        if self._options.extension == 'DRM':
            if self._options.stats and name == "drm_status_cb":
                self._process_ext_msg(False, None, value)
            elif name != "drm_status_cb" and name != "drm_bar_pct" and name != "annotate":
                self._process_ext_msg(True, name, value)
            if name == "locked" and value != "1":
                raise Exception("No DRM when Kiwi running other extensions or too many connections active")
        else:
            self._process_ext_msg(True, name, value)

## -------------------------------------------------------------------------------------------------

class KiwiNetcat(KiwiSDRStream):
    def __init__(self, options, reader):
        super(KiwiNetcat, self).__init__()
        _init_common(self, options, 'W/F' if options.waterfall is True else 'SND')
        self._reader = reader
        self._start_time = time.time()
        self._options.stats = None
        self._fp_stdout = os.fdopen(sys.stdout.fileno(), 'wb')
        self._first = True

    def _setup_rx_params(self):
        user = self._options.user

        if self._camp_chan != -1 and self._kiwi_version is not None and self._kiwi_version < 1.813:
            raise Exception('Option --camp requires camped Kiwi be running v1.813 or later')

        if self._type == 'SND':
            _setup_rx_snd_params(self, user)
        else:   # waterfall
            self.set_name(user)
            self._set_maxdb_mindb(-10, -110)    # needed, but values don't matter
            self._set_zoom_cf(0, 0)
            self._set_wf_comp(self._options.wf_comp)
            self._set_wf_speed(1)   # 1 Hz update

    def _process_audio_samples(self, seq, samples, rssi, fmt):
        if self._options.progress is True:
            fmt = "" if fmt is None or self._options.log_level != 'debug' else fmt
            sys.stderr.write('\rBlock: %08x, RSSI: %6.1f %s ' % (seq, rssi, fmt))
            sys.stderr.flush()

        if self._options.resample > 0:
            if HAS_RESAMPLER:
                ## libsamplerate resampling
                if self._resampler is None:
                    self._resampler = Resampler(converter_type='sinc_best')
                    _setup_resampler(self)
                samples = np.round(self._resampler.process(samples, ratio=self._ratio)).astype(np.int16)
            else:
                ## resampling by linear interpolation
                n  = len(samples)
                xa = np.arange(round(n*self._ratio))/self._ratio
                xp = np.arange(n)
                samples = np.round(np.interp(xa,xp,samples)).astype(np.int16)

        self._write_samples(samples, {})

    def _process_iq_samples(self, seq, samples, rssi, gps, fmt):
        if self._options.progress is True:
            fmt = "" if fmt is None or self._options.log_level != 'debug' else fmt
            sys.stderr.write('\rBlock: %08x, RSSI: %6.1f %s ' % (seq, rssi, fmt))
            sys.stderr.flush()

        if self._camping or self._options.resample > 0:
            ## convert list of complex numbers into an array
            s = np.zeros(2*len(samples), dtype=np.int16)
            s[0::2] = np.real(samples).astype(np.int16)
            s[1::2] = np.imag(samples).astype(np.int16)
        else:
            s = samples

        if self._options.resample > 0:
            if HAS_RESAMPLER:
                ## libsamplerate resampling
                if self._resampler is None:
                    self._resampler = Resampler(channels=2, converter_type='sinc_best')
                    _setup_resampler(self)
                s = self._resampler.process(s.reshape(len(samples),2), ratio=self._ratio)
                s = np.round(s.flatten()).astype(np.int16)
            else:
                ## resampling by linear interpolation
                n  = len(samples)
                m  = int(round(n*self._ratio))
                xa = np.arange(m)/self._ratio
                xp = np.arange(n)
                s  = np.zeros(2*m, dtype=np.int16)
                s[0::2] = np.round(np.interp(xa,xp,np.real(samples))).astype(np.int16)
                s[1::2] = np.round(np.interp(xa,xp,np.imag(samples))).astype(np.int16)

        self._write_samples(s, {})

    def _process_waterfall_samples_raw(self, seq, samples):
        if self._options.progress is True:
            nbins = len(samples)
            bins = nbins-1
            max = -1
            min = 256
            bmax = bmin = 0
            i = 0
            for s in samples:
                if s > max:
                    max = s
                    bmax = i
                if s < min:
                    min = s
                    bmin = i
                i += 1
            span = 30000
            sys.stderr.write('\rwf samples %d bins %d..%d dB %.1f..%.1f kHz rbw %d kHz'
                  % (nbins, min-255, max-255, span*bmin/bins, span*bmax/bins, span/bins))
            sys.stderr.flush()
        self._fp_stdout.write(samples)
        self._fp_stdout.flush()

    def _write_samples(self, samples, *args):
        if self._options.nc_wav and self._first == True:
            _write_wav_header(self, self._fp_stdout, 0x7ffffff0, self._output_sample_rate, self._num_channels, False)
            self._first = False
        self._fp_stdout.write(samples)
        self._fp_stdout.flush()

    def _writer_message(self):
        if not self._options.fdx:
            time.sleep(1)
            return None
        
        test = 0
        #test = 1
        
        if self._options.writer_init == False:
            self._options.writer_init = True
            self._fdx_seq = 0
            self._q_readline = Queue()
            t = threading.Thread(target=enqueue_input, args=(sys.stdin, self._q_readline))
            t.daemon = True     # thread dies with the program
            t.start()
            return 'SET msg_log=camp_fdx_setup' if test else None
        
        if test:
            self._fdx_seq = self._fdx_seq + 1
            msg = 'SET msg_log=camp_fdx_test_%d' % self._fdx_seq
            time.sleep(0.2)
        else:
            # stackoverflow.com/questions/375427/a-non-blocking-read-on-a-subprocess-pipe-in-python
            try:
                #line = self._q_readline.get_nowait()
                line = self._q_readline.get(timeout=0.1)
                time.sleep(0.1)     # throttle, because the above return immediately if no data
            except Empty:
                msg = None
                time.sleep(0.1)
            else:
                if line == None or line == '':
                    msg = None
                else:
                    msg = line
                    #logging.debug('====> %s' % msg)
        return msg

## -------------------------------------------------------------------------------------------------

def options_cross_product(options):
    """build a list of options according to the number of servers specified"""
    def _sel_entry(i, l):
        """if l is a list, return the element with index i, else return l"""
        return l[min(i, len(l)-1)] if type(l) == list else l

    l = []
    multiple_connections = 0
    for i,s in enumerate(options.server_host):
        opt_single = copy(options)
        opt_single.server_host = s
        opt_single.status = 0

        # time() returns seconds, so add pid and host index to make timestamp unique per connection
        opt_single.ws_timestamp = int(time.time() + os.getpid() + i) & 0xffffffff
        for x in ['server_port', 'password', 'tlimit_password', 'frequency', 'agc_gain', 'filename', 'station', 'user']:
            opt_single.__dict__[x] = _sel_entry(i, opt_single.__dict__[x])
        l.append(opt_single)
        multiple_connections = i
    return multiple_connections,l

def get_comma_separated_args(option, opt, value, parser, fn):
    values = [fn(v.strip()) for v in value.split(',')]
    setattr(parser.values, option.dest, values)
##    setattr(parser.values, option.dest, map(fn, value.split(',')))

def join_threads(snd, wf, ext, nc):
    [r._event.set() for r in snd]
    [r._event.set() for r in wf]
    [r._event.set() for r in ext]
    [r._event.set() for r in nc]
    [t.join() for t in threading.enumerate() if t is not threading.current_thread()]

def main():
    # extend the OptionParser so that we can print multiple paragraphs in
    # the help text
    class MyParser(OptionParser):
        def format_description(self, formatter):
            result = []
            for paragraph in self.description:
                result.append(formatter.format_description(paragraph))
            return "\n".join(result[:-1]) # drop last \n

        def format_epilog(self, formatter):
            result = []
            for paragraph in self.epilog:
                result.append(formatter.format_epilog(paragraph))
            return "".join(result)

    usage = "%prog -s SERVER -p PORT -f FREQ -m MODE [other options]"
    description = ["kiwirecorder.py records data from one or more KiwiSDRs to your disk."
                   " It takes a number of options as inputs, the most basic of which"
                   " are shown above.",
                   "To record data from multiple Kiwis at once, use the same syntax,"
              " but pass a list of values (where applicable) instead of a single value."
              " Each list of values should be comma-separated and without spaces."
              " For instance, to record one Kiwi at localhost on port 80, and another Kiwi"
              " at example.com port 8073, run the following:",
              "    kiwirecorder.py -s localhost,example.com -p 80,8073 -f 10000,10000 -m am",
              "In this example, both Kiwis will record on 10,000 kHz (10 MHz) in AM mode."
              " Any option that states \"can be a comma-separated list\" also means a single"
              " value will be duplicated across multiple connection. In the above example"
              " the simpler \"-f 10000\" can been used.", ""]
    epilog = [] # text here would go after the options list
    parser = MyParser(usage=usage, description=description, epilog=epilog)
    parser.add_option('-s', '--server-host',
                      dest='server_host',
                      type='string', default='localhost',
                      help='Server host (can be a comma-separated list)',
                      action='callback',
                      callback_args=(str,),
                      callback=get_comma_separated_args)
    parser.add_option('-p', '--server-port',
                      dest='server_port',
                      type='string', default=8073,
                      help='Server port, default 8073 (can be a comma-separated list)',
                      action='callback',
                      callback_args=(int,),
                      callback=get_comma_separated_args)
    parser.add_option('--pw', '--password',
                      dest='password',
                      type='string', default='',
                      help='Kiwi login password (if required, can be a comma-separated list)',
                      action='callback',
                      callback_args=(str,),
                      callback=get_comma_separated_args)
    parser.add_option('--admin',
                      dest='admin',
                      action='store_true', default=False,
                      help='Kiwi login as admin')
    parser.add_option('--tlimit-pw', '--tlimit-password',
                      dest='tlimit_password',
                      type='string', default='',
                      help='Connect time limit exemption password (if required, can be a comma-separated list)',
                      action='callback',
                      callback_args=(str,),
                      callback=get_comma_separated_args)
    parser.add_option('-u', '--user',
                      dest='user',
                      type='string', default='kiwirecorder.py',
                      help='Kiwi connection user name (can be a comma-separated list)',
                      action='callback',
                      callback_args=(str,),
                      callback=get_comma_separated_args)
    parser.add_option('--station',
                      dest='station',
                      type='string', default=None,
                      help='Station ID to be appended to filename (can be a comma-separated list)',
                      action='callback',
                      callback_args=(str,),
                      callback=get_comma_separated_args)
    parser.add_option('--log', '--log-level', '--log_level',
                      dest='log_level',
                      type='choice', default='warn',
                      choices=['debug', 'info', 'warn', 'error', 'critical'],
                      help='Log level: debug|info|warn(default)|error|critical')
    parser.add_option('-q', '--quiet',
                      dest='quiet',
                      action='store_true', default=False,
                      help='Don\'t print progress messages')
    parser.add_option('--nq', '--not-quiet',
                      dest='not_quiet',
                      action='store_true', default=False,
                      help='Print progress messages')
    parser.add_option('-d', '--dir',
                      dest='dir',
                      type='string', default=None,
                      help='Optional destination directory for files')
    parser.add_option('--fn', '--filename',
                      dest='filename',
                      type='string', default='',
                      help='Use fixed filename instead of generated filenames (optional station ID(s) will apply, can be a comma-separated list)',
                      action='callback',
                      callback_args=(str,),
                      callback=get_comma_separated_args)
    parser.add_option('--tlimit', '--time-limit',
                      dest='tlimit',
                      type='float', default=None,
                      help='Record time limit in seconds. Ignored when --dt-sec used.')
    parser.add_option('--dt-sec',
                      dest='dt',
                      type='int', default=0,
                      help='Start a new file when mod(sec_of_day,dt) == 0')
    parser.add_option('--launch-delay', '--launch_delay',
                      dest='launch_delay',
                      type='int', default=0,
                      help='Delay (secs) in launching multiple connections')
    parser.add_option('--connect-timeout', '--connect_timeout',
                      dest='connect_timeout',
                      type='int', default=15,
                      help='Retry timeout(sec) connecting to host')
    parser.add_option('--connect-retries', '--connect_retries',
                      dest='connect_retries',
                      type='int', default=0,
                      help='Number of retries when connecting to host (retries forever by default)')
    parser.add_option('--busy-timeout', '--busy_timeout',
                      dest='busy_timeout',
                      type='int', default=15,
                      help='Retry timeout(sec) when host is busy')
    parser.add_option('--busy-retries', '--busy_retries',
                      dest='busy_retries',
                      type='int', default=0,
                      help='Number of retries when host is busy (retries forever by default)')
    parser.add_option('-k', '--socket-timeout', '--socket_timeout',
                      dest='socket_timeout',
                      type='int', default=10,
                      help='Socket timeout(sec) during data transfers')
    parser.add_option('--OV',
                      dest='ADC_OV',
                      action='store_true', default=False,
                      help='Print "ADC OV" message when Kiwi ADC is overloaded')
    parser.add_option('--ts', '--tstamp', '--timestamp',
                      dest='tstamp',
                      action='store_true', default=False,
                      help='Add timestamps to output. Applies only to S-meter mode currently.')
    parser.add_option('--stats',
                      dest='stats',
                      action='store_true', default=False,
                      help='Print additional statistics. Applies to e.g. S-meter and extension modes.')
    parser.add_option('-v', '-V', '--version',
                      dest='krec_version',
                      action='store_true', default=False,
                      help='Print version number and exit')

    group = OptionGroup(parser, "Audio connection options", "")
    group.add_option('-f', '--freq',
                      dest='frequency',
                      type='string', default=15000,     # 15000 prevents --wf mode span error for zoom=0
                      help='Frequency to tune to, in kHz (can be a comma-separated list). '
                        'For sideband modes (lsb/lsn/usb/usn/cw/cwn) this is the carrier/dial frequency. '
                        'See --pbc option below. Also sets waterfall mode center frequency.',
                      action='callback',
                      callback_args=(float,),
                      callback=get_comma_separated_args)
    group.add_option('--pbc', '--freq-pbc',
                      dest='freq_pbc',
                      action='store_true', default=False,
                      help='For sideband modes (lsb/lsn/usb/usn/cw/cwn) interpret -f/--freq frequency as the passband center frequency.')
    group.add_option('-o', '--offset', '--foffset',
                      dest='freq_offset',
                      type='int', default=0,
                      help='Frequency offset (kHz) subtracted from tuned frequency (for those Kiwis using an offset)')
    group.add_option('-m', '--mode', '--modulation',
                      dest='modulation',
                      type='string', default='am',
                      help='Modulation; one of am/amn/amw, sam/sau/sal/sas/qam, lsb/lsn, usb/usn, cw/cwn, nbfm/nnfm, iq (default passband if -L/-H not specified)')
    group.add_option('--ncomp', '--no_compression', '--no_compression',
                      dest='compression',
                      action='store_false', default=True,
                      help='Don\'t use audio compression (IQ mode never uses compression)')
    group.add_option('-L', '--lp-cutoff',
                      dest='lp_cut',
                      type='float', default=None,
                      help='Low-pass cutoff frequency, in Hz')
    group.add_option('-H', '--hp-cutoff',
                      dest='hp_cut',
                      type='float', default=None,
                      help='High-pass cutoff frequency, in Hz')
    group.add_option('-r', '--resample',
                      dest='resample',
                      type='int', default=0,
                      help='Resample output file to new sample rate in Hz. The resampling ratio has to be in the range [1/256,256]')
    group.add_option('-T', '--squelch-threshold',
                      dest='sq_thresh',
                      type='float', default=None,
                      help='Squelch threshold, in dB.')
    group.add_option('--squelch-tail',
                      dest='squelch_tail',
                      type='float', default=1,
                      help='Time for which the squelch remains open after the signal is below threshold.')
    group.add_option('-g', '--agc-gain',
                      dest='agc_gain',
                      type='string', default=None,
                      help='AGC gain; if set, AGC is turned off (can be a comma-separated list)',
                      action='callback',
                      callback_args=(float,),
                      callback=get_comma_separated_args)
    group.add_option('--agc-yaml',
                      dest='agc_yaml_file',
                      type='string', default=None,
                      help='AGC options provided in a YAML-formatted file')
    group.add_option('--scan-yaml',
                      dest='scan_yaml_file',
                      type='string', default=None,
                      help='Scan options provided in a YAML-formatted file')
    group.add_option('--nb',
                      dest='nb',
                      action='store_true', default=False,
                      help='Enable standard noise blanker with default parameters.')
    group.add_option('--nb-gate',
                      dest='nb_gate',
                      type='int', default=100,
                      help='Noise blanker gate time in usec (100 to 5000, default 100)')
    group.add_option('--nb-th', '--nb-thresh',
                      dest='nb_thresh',
                      type='int', default=50,
                      help='Noise blanker threshold in percent (0 to 100, default 50)')
    group.add_option('--nb-test',
                      dest='nb_test',
                      action='store_true', default=False,
                      help='Enable noise blanker test mode.')
    group.add_option('--de-emp',
                      dest='de_emp',
                      action='store_true', default=False,
                      help='Enable de-emphasis.')
    group.add_option('-w', '--kiwi-wav',
                      dest='is_kiwi_wav',
                      action='store_true', default=False,
                      help='In the wav file include KIWI header containing GPS time-stamps (only for IQ mode)')
    group.add_option('--kiwi-tdoa',
                      dest='is_kiwi_tdoa',
                      action='store_true', default=False,
                      help='Used when called by Kiwi TDoA extension')
    group.add_option('--test-mode', '--no-output-files',
                      dest='test_mode',
                      action='store_true', default=False,
                      help='Write wav data to /dev/null (Linux) or NUL (Windows)')
    group.add_option('--snd', '--sound',
                      dest='sound',
                      action='store_true', default=False,
                      help='Also process sound data when in waterfall or S-meter mode (sound connection options above apply)')
    group.add_option('--camp', '--camp-chan',
                      dest='camp_chan',
                      type='int', default=-1,
                      help='Camp on an existing audio channel instead of opening a new connection. Argument is Kiwi channel number. Note that camping does not currently support resampling, squelch or GPS timestamps.')
    group.add_option('--wb', '--wideband',
                      dest='wideband',
                      action='store_true', default=False,
                      help='Open a wideband connection to Kiwi (if supported)')
    parser.add_option_group(group)

    group = OptionGroup(parser, "S-meter mode options", "")
    group.add_option('--S-meter', '--s-meter',
                      dest='S_meter',
                      type='int', default=-1,
                      help='Report S-meter (RSSI) value after S_METER number of averages. S_METER=0 does no averaging and reports each RSSI value received. Options --ts and --stats apply.')
    group.add_option('--sdt-sec',
                      dest='sdt',
                      type='int', default=0,
                      help='S-meter measurement interval')
    parser.add_option_group(group)

    group = OptionGroup(parser, "Waterfall connection options", "")
    group.add_option('--wf', '--waterfall',
                      dest='waterfall',
                      action='store_true', default=False,
                      help='Process waterfall data instead of audio. Center frequency set by audio option --f/--freq')
    group.add_option('-z', '--zoom',
                      dest='zoom',
                      type='int', default=0,
                      help='Zoom level 0-14')
    group.add_option('--speed',
                      dest='speed',
                      type='int', default=0,
                      help='Waterfall update speed: 1=1Hz, 2=slow, 3=med, 4=fast')
    group.add_option('--interp', '--wf-interp',
                      dest='interp',
                      type='int', default=-1,
                      help='Waterfall display interpolation 0-13')
    group.add_option('--wf-png',
                      dest='wf_png',
                      action='store_true', default=False,
                      help='Create waterfall .png file. --station and --filename options apply')
    group.add_option('--wf-peaks',
                      dest='wf_peaks',
                      type='int', default=0,
                      help='Save specified number of waterfall peaks to file. --station and --filename options apply')
    group.add_option('--maxdb',
                      dest='maxdb',
                      type='int', default=-30,
                      help='Waterfall colormap max dB (-170 to -10)')
    group.add_option('--mindb',
                      dest='mindb',
                      type='int', default=-155,
                      help='Waterfall colormap min dB (-190 to -30)')
    group.add_option('--wf-auto',
                      dest='wf_auto',
                      action='store_true', default=False,
                      help='Auto set mindb/maxdb')
    group.add_option('--wf-cal',
                      dest='wf_cal',
                      type='int', default=None,
                      help='Waterfall calibration correction (overrides Kiwi default value)')
    group.add_option('--wf-comp',
                      dest='wf_comp',
                      action='store_true', default=False,
                      help='Enable waterfall compression')
    group.add_option('--snr',
                      dest='snr',
                      type='int', default=0,
                      help='Compute SNR value after specified number of waterfall averages')
    parser.add_option_group(group)

    group = OptionGroup(parser, "DX label control options", "")
    group.add_option('--dx', '--dx-list',
                      dest='dx_list',
                      type='float', nargs=2, default=None,
                      help='List DX labels. DX_LIST is MIN MAX freqs in kHz. For example "--dx=2500 5000". Results displayed and written to file named "dx.json"')
    group.add_option('--dx-add',
                      dest='dx_add',
                      action='store_true', default=False,
                      help='Add a new DX label. See Makefile for example')
    group.add_option('--dx-del',
                      dest='dx_del',
                      type='int', default=None,
                      help='Delete a DX label specified by DX_DEL number, which is shown in the output from --dx-list. CAUTION: This number changes as DX list content changes. Always inspect latest --dx-list before deleting an entry.')
    group.add_option('--dx-type',
                      dest='dx_type',
                      type='int', default=0,
                      help='DX label type (0-15) to use with --dx-add')
    group.add_option('--dx-ident',
                      dest='dx_ident',
                      type='string', default='ident',
                      help='DX label ident string to use with --dx-add')
    group.add_option('--dx-notes',
                      dest='dx_notes',
                      type='string', default='',
                      help='DX label notes string to use with --dx-add')
    group.add_option('--dx-params',
                      dest='dx_params',
                      type='string', default='',
                      help='DX label parameter string to use with --dx-add')
    parser.add_option_group(group)

    group = OptionGroup(parser, "Extension connection options", "")
    group.add_option('--ext',
                      dest='extension',
                      type='string', default=None,
                      help='Also open a connection to EXTENSION name')
    group.add_option('--ext-test',
                      dest='ext_test',
                      action='store_true', default=False,
                      help='Start extension in its test mode (if applicable)')
    parser.add_option_group(group)

    group = OptionGroup(parser, "Netcat connection options", "")
    group.add_option('--nc', '--netcat',
                      dest='netcat',
                      action='store_true', default=False,
                      help='Open a netcat connection. Note that netcat does not currently support resampling, squelch or GPS timestamps.')
    group.add_option('--nc-wav', '--nc_wav',
                      dest='nc_wav',
                      action='store_true', default=False,
                      help='Format output as an continuous wav file stream. Note that --nc-wav does not currently support resampling, squelch or GPS timestamps.')
    group.add_option('--fdx',
                      dest='fdx',
                      action='store_true', default=False,
                      help='Connection is full duplex (two-way)')
    group.add_option('--progress',
                      dest='progress',
                      action='store_true', default=False,
                      help='Print progress messages (doesn\'t effect streamed data)')
    parser.add_option_group(group)

    group = OptionGroup(parser, "KiwiSDR development options", "")
    group.add_option('--gc-stats',
                      dest='gc_stats',
                      action='store_true', default=False,
                      help='Print garbage collection stats')
    group.add_option('--nolocal',
                      dest='nolocal',
                      action='store_true', default=False,
                      help='Make local network connections appear non-local')
    group.add_option('--no-api',
                      dest='no_api',
                      action='store_true', default=False,
                      help='Simulate connection to Kiwi using improper/incomplete API')
    group.add_option('--bad-cmd',
                      dest='bad_cmd',
                      action='store_true', default=False,
                      help='Simulate connection to Kiwi using improper/incomplete API')
    group.add_option('--devel',
                      dest='devel',
                      type='string', default=None,
                      help='Set development parameters p0-p7 to float value. Format: [0-7]:float_value, ...')
    parser.add_option_group(group)

    opts_no_defaults = optparse.Values()
    __, args = parser.parse_args(values=opts_no_defaults)
    options = optparse.Values(parser.get_default_values().__dict__)
    options._update_careful(opts_no_defaults.__dict__)

    ## clean up OptionParser which has cyclic references
    parser.destroy()

    if options.krec_version:
        print('kiwirecorder %s' % VERSION)
        sys.exit()

    FORMAT = '%(asctime)-15s pid %(process)5d %(message)s'
    logging.basicConfig(level=logging.getLevelName(options.log_level.upper()), format=FORMAT)
    if options.gc_stats:
        gc.set_debug(gc.DEBUG_SAVEALL | gc.DEBUG_LEAK | gc.DEBUG_UNCOLLECTABLE)

    run_event = threading.Event()
    run_event.set()

    if options.freq_pbc:
        logging.info('command line: --pbc')

    if options.S_meter >= 0:
        if options.S_meter > 0 and options.sdt != 0:
            raise Exception('Options --S-meter > 0 and --sdt-sec != 0 are incompatible. Did you mean to use --S-meter=0 ?')
        options.quiet = True

    if options.tlimit is not None and options.dt != 0:
        print('Warning: --tlimit ignored when --dt-sec option used')

    if options.wf_png is True:
        if options.waterfall is False:
            options.waterfall = True
            print('--wf-png note: assuming --wf')
        if options.speed == 0:
            options.speed = 4
            print('--wf-png note: no --speed specified, so using fast (=4)')
        options.quite = True    # specify "--not-quiet" to see all progress messages during --wf-png

    if options.wf_peaks > 0:
        if options.interp == -1:
            options.interp = 10
            print('--wf-peaks note: no --wf-interp specified, so using MAX+CIC (=10)')

    if options.dx_list is not None or options.dx_add or options.dx_del != None:
        options.waterfall = True
        options.zoom = 14
        options.dx = True
    
    if (options.dx_add or options.dx_del) and not options.admin:
        raise Exception('--dx-add and --dx-del require --admin option, and possibly admin password via --pw, to be given.')

    if options.netcat:
        if options.sq_thresh is not None:
            raise Exception('squelch not currently supported by --netcat. Email Kiwi support if you really need it.')
        if options.is_kiwi_wav is True:
            raise Exception('GPS timestamps not currently supported by --netcat. Email Kiwi support if you really need it.')

    ### decode AGC YAML file options
    options.agc_yaml = None
    if options.agc_yaml_file:
        try:
            if not HAS_PyYAML:
                raise Exception('PyYAML not installed: sudo apt install python-yaml / sudo apt install python3-yaml / pip install pyyaml / pip3 install pyyaml')
            with open(options.agc_yaml_file) as yaml_file:
                documents = yaml.full_load(yaml_file)
                logging.debug('AGC file %s: %s' % (options.agc_yaml_file, documents))
                logging.debug('Got AGC parameters from file %s: %s' % (options.agc_yaml_file, documents['AGC']))
                options.agc_yaml = documents['AGC']
        except KeyError:
            logging.fatal('The YAML file does not contain AGC options')
            return
        except Exception as e:
            logging.fatal(e)
            return

    ### decode scan YAML file options
    options.scan_yaml = None
    if options.scan_yaml_file:
        try:
            if not HAS_PyYAML:
                raise Exception('PyYAML not installed: sudo apt install python-yaml / sudo apt install python3-yaml / pip install pyyaml / pip3 install pyyaml')
            if hasattr(opts_no_defaults, 'frequency'):
                raise Exception('cannot specify frequency (-f, --freq) together with scan YAML (--scan-yaml)')
            with open(options.scan_yaml_file) as yaml_file:
                documents = yaml.full_load(yaml_file)
                logging.debug('Scan file %s: %s' % (options.scan_yaml_file, documents))
                logging.debug('Got Scan parameters from file %s: %s' % (options.scan_yaml_file, documents['Scan']))
                options.scan_yaml = documents['Scan']
                options.scan_time = time.time()
                options.scan_index = 0
                options.scan_yaml['frequencies'] = [float(f) for f in options.scan_yaml['frequencies']]
                options.frequency = options.scan_yaml['frequencies'][0]
        except KeyError:
            options.scan_yaml = None
            logging.fatal('The YAML file does not contain Scan options')
            return
        except Exception as e:
            options.scan_yaml = None
            logging.fatal(e)
            return

    options.rigctl_enabled = False
    
    options.maxdb = clamp(options.maxdb, -170, -10)
    options.mindb = clamp(options.mindb, -190, -30)
    if options.maxdb <= options.mindb:
        options.maxdb = options.mindb + 1

    gopt = options
    multiple_connections,options = options_cross_product(options)

    snd_recorders = []
    if not gopt.netcat and (not gopt.waterfall or (gopt.waterfall and gopt.sound)):
        for i,opt in enumerate(options):
            opt.multiple_connections = multiple_connections
            opt.idx = i
            snd_recorders.append(KiwiWorker(args=(KiwiSoundRecorder(opt),opt,True,False,run_event)))

    wf_recorders = []
    if not gopt.netcat and gopt.waterfall:
        for i,opt in enumerate(options):
            opt.multiple_connections = multiple_connections
            opt.idx = i
            wf_recorders.append(KiwiWorker(args=(KiwiWaterfallRecorder(opt),opt,True,False,run_event)))

    ext_recorders = []
    if not gopt.netcat and (gopt.extension is not None):
        for i,opt in enumerate(options):
            opt.multiple_connections = multiple_connections
            opt.idx = i
            ext_recorders.append(KiwiWorker(args=(KiwiExtensionRecorder(opt),opt,True,True,run_event)))

    nc_recorders = []
    if gopt.netcat:
        for i,opt in enumerate(options):
            opt.multiple_connections = multiple_connections
            opt.idx = 0
            nc_recorders.append(KiwiWorker(args=(KiwiNetcat(opt, True),opt,True,False,run_event)))
            if gopt.fdx:
                opt.idx = 1
                nc_recorders.append(KiwiWorker(args=(KiwiNetcat(opt, False),opt,False,False,run_event)))
    try:
        for i,r in enumerate(snd_recorders):
            if gopt.launch_delay != 0 and i != 0 and options[i-1].server_host == options[i].server_host:
                time.sleep(gopt.launch_delay)
            r.start()
            #logging.info("started sound recorder %d, timestamp=%d" % (i, options[i].ws_timestamp))
            logging.info("started sound recorder %d" % i)

        for i,r in enumerate(wf_recorders):
            if gopt.launch_delay != 0 and i != 0 and options[i-1].server_host == options[i].server_host:
                time.sleep(gopt.launch_delay)
            r.start()
            logging.info("started waterfall recorder %d" % i)

        for i,r in enumerate(ext_recorders):
            if gopt.launch_delay != 0 and i != 0 and options[i-1].server_host == options[i].server_host:
                time.sleep(gopt.launch_delay)
            r.start()
            logging.info("started extension recorder %d" % i)

        for i,r in enumerate(nc_recorders):
            if gopt.launch_delay != 0 and i != 0 and options[i-1].server_host == options[i].server_host:
                time.sleep(gopt.launch_delay)
            r.start()
            #logging.info("started netcat recorder %d, timestamp=%d" % (i, options[i].ws_timestamp))
            logging.info("started netcat recorder %d" % i)

        while run_event.is_set():
            time.sleep(.1)

    except KeyboardInterrupt:
        run_event.clear()
        join_threads(snd_recorders, wf_recorders, ext_recorders, nc_recorders)
        print("KeyboardInterrupt: threads successfully closed")
    except Exception as e:
        print_exc()
        run_event.clear()
        join_threads(snd_recorders, wf_recorders, ext_recorders, nc_recorders)
        print("Exception: threads successfully closed")

    if gopt.is_kiwi_tdoa:
      for i,opt in enumerate(options):
          # NB for TDoA support: MUST be a print to stdout (i.e. not a logging.info to stderr)
          print("status=%d,%d" % (i, opt.status))

    if gopt.gc_stats:
        logging.debug('gc %s' % gc.garbage)

if __name__ == '__main__':
    #import faulthandler
    #faulthandler.enable()
    main()
# EOF
