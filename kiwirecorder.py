#!/usr/bin/env python
## -*- python -*-

import array, logging, os, struct, sys, time, copy, threading, os
import gc
import math
import numpy as np
from copy import copy
from traceback import print_exc
from kiwi import KiwiSDRStream, KiwiWorker
import optparse as optparse
from optparse import OptionParser
from optparse import OptionGroup

HAS_PyYAML = True
try:
    ## needed for the --agc-yaml option
    import yaml
    if yaml.__version__.split('.')[0] < '5':
        print('wrong PyYAML version: %s < 5; PyYAML is only needed when using the --agc-yaml option' % yaml.__version__)
        raise ImportError
except ImportError:
    ## (only) when needed an exception is raised, see below
    HAS_PyYAML = False

HAS_RESAMPLER = True
try:
    ## if available use libsamplerate for resampling
    from samplerate import Resampler
except ImportError:
    ## otherwise linear interpolation is used
    HAS_RESAMPLER = False

try:
    if os.environ['USE_LIBSAMPLERATE'] == 'False':
        HAS_RESAMPLER = False
except KeyError:
    pass

def _write_wav_header(fp, filesize, samplerate, num_channels, is_kiwi_wav):
    fp.write(struct.pack('<4sI4s', b'RIFF', filesize - 8, b'WAVE'))
    bits_per_sample = 16
    byte_rate       = samplerate * num_channels * bits_per_sample // 8
    block_align     = num_channels * bits_per_sample // 8
    fp.write(struct.pack('<4sIHHIIHH', b'fmt ', 16, 1, num_channels, int(samplerate+0.5), byte_rate, block_align, bits_per_sample))
    if not is_kiwi_wav:
        fp.write(struct.pack('<4sI', b'data', filesize - 12 - 8 - 16 - 8))

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

            logging.info('%s: (%2d,%3d) t_gnss= %16.9f %s'
                         % (filename, self._last_solution, self._num_frames, ts, msg_gnss_drift))
            self._num_frames = 0
            self._last_ts    = ts

        self._last_solution = gps['last_gps_solution']


class Squelch(object):
    def __init__(self, options):
        self._status_msg  = not options.quiet
        self._threshold   = options.sq_thresh
        self._squelch_tail = options.squelch_tail ## in seconds
        self._ring_buffer = RingBuffer(65)
        self._squelch_on_seq = None
        self.set_sample_rate(12000.0) ## default setting

    def set_threshold(self, threshold):
        self._threshold = threshold
        return self

    def set_sample_rate(self, fs):
        self._tail_delay  = round(self._squelch_tail*fs/512) ## seconds to number of buffers

    def process(self, seq, rssi):
        if not self._ring_buffer.is_filled() or self._squelch_on_seq is None:
            self._ring_buffer.insert(rssi)
        if not self._ring_buffer.is_filled():
            return False
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
            sys.stdout.write('\r Median: %6.1f Thr: %6.1f %s' % (median_nf, rssi_thresh, ("s", "S")[is_open]))
            sys.stdout.flush()
        if not is_open:
            return False
        if seq > self._squelch_on_seq + self._tail_delay:
            logging.info("\nSquelch closed")
            self._squelch_on_seq = None
            return False
        return is_open

class KiwiSoundRecorder(KiwiSDRStream):
    def __init__(self, options):
        super(KiwiSoundRecorder, self).__init__()
        self._options = options
        self._type = 'SND'
        freq = options.frequency
        #logging.info("%s:%s freq=%d" % (options.server_host, options.server_port, freq))
        self._freq = freq
        self._start_ts = None
        self._start_time = None
        self._squelch = Squelch(self._options) if options.sq_thresh is not None else None
        if options.scan_yaml is not None:
            self._squelch = [Squelch(options).set_threshold(options.scan_yaml['threshold']) for _ in range(len(options.scan_yaml['frequencies']))]
        self._last_gps = dict(zip(['last_gps_solution', 'dummy', 'gpssec', 'gpsnsec'], [0,0,0,0]))
        self._resampler = None
        self._gnss_performance = GNSSPerformance()

    def set_freq(self, freq):
        self._freq = freq
        mod    = self._options.modulation
        lp_cut = self._options.lp_cut
        hp_cut = self._options.hp_cut
        if mod == 'am' or mod == 'amn':
            # For AM, ignore the low pass filter cutoff
            lp_cut = -hp_cut if hp_cut is not None else hp_cut
        if mod == 'lsb':
            # For LSB, transpose and invert the cutoffs
            lp_cut2 = -hp_cut
            hp_cut2 = -lp_cut
            hp_cut = hp_cut2
            lp_cut = lp_cut2
        self.set_mod(mod, lp_cut, hp_cut, self._freq)

    def _setup_rx_params(self):
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

        if self._options.nb is True:
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
                    s.set_sample_rate(self._sample_rate)
            else:
                self._squelch.set_sample_rate(self._sample_rate)

        if self._options.test_mode:
            self._set_stats()

        if self._options.resample > 0:
            if not HAS_RESAMPLER:
                self._output_sample_rate = self._options.resample
                self._ratio = float(self._output_sample_rate)/self._sample_rate
                logging.info("libsamplerate not available: linear interpolation is used for low-quality resampling. "
                             "(pip/pip3 install samplerate)")
                logging.info('resampling from %g to %d Hz (ratio=%f)' % (self._sample_rate, self._options.resample, self._ratio))
            else:
                fs = 10*round(self._sample_rate/10) ## rounded sample rate
                ratio = self._options.resample / fs
                ## work around a bug in python-libsamplerate:
                ##  the following makes sure that ratio * 512 is an integer
                ##  at the expense of resampling frequency precision for some resampling frequencies (it's ok for 375 Hz)
                n = 512 ## KiwiSDR block length for samples
                m = round(ratio*n)
                self._ratio = m/n
                self._output_sample_rate = self._ratio * self._sample_rate
                logging.info('resampling from %g to %g Hz (ratio=%f)' % (self._sample_rate, self._output_sample_rate, self._ratio))

    def _squelch_status(self, seq, samples, rssi):
        if not self._options.quiet:
            sys.stdout.write('\rBlock: %08x, RSSI: %6.1f' % (seq, rssi))
        if self._squelch and type(self._squelch) == list: ## scan mode
            if self._options.quiet:
                sys.stdout.write('\r')
            sys.stdout.write(" scan: [%s] freq = %g kHz      " % (self._options.scan_state, self._freq))
        sys.stdout.flush()

        is_open = True
        if self._squelch:
            if type(self._squelch) == list: ## scan mode
                if self._options.scan_state == "WAIT":
                    is_open = False
                    now = time.time()
                    if now - self._options.scan_time > self._options.scan_yaml['wait']:
                        self._options.scan_time = now
                        self._options.scan_state = 'DWELL'
                if self._options.scan_state == 'DWELL':
                    is_open = self._squelch[self._options.scan_index].process(seq, rssi)
                    now = time.time()
                    if not is_open and now - self._options.scan_time > self._options.scan_yaml['dwell']:
                        self._options.scan_index = (self._options.scan_index + 1) % len(self._options.scan_yaml['frequencies'])
                        self.set_freq(self._options.scan_yaml['frequencies'][self._options.scan_index])
                        self._options.scan_time = now
                        self._options.scan_state = 'WAIT'
                        self._start_ts = None
                        self._start_time = None
            else: ## single channel mode
                is_open = self._squelch.process(seq, rssi)
                if not is_open:
                    self._start_ts = None
                    self._start_time = None
        return is_open


    def _process_audio_samples(self, seq, samples, rssi):
        is_open = self._squelch_status(seq, samples, rssi)
        if not is_open:
            return


        if self._options.resample > 0:
            if HAS_RESAMPLER:
                ## libsamplerate resampling
                if self._resampler is None:
                    self._resampler = Resampler(converter_type='sinc_best')
                samples = np.round(self._resampler.process(samples, ratio=self._ratio)).astype(np.int16)
            else:
                ## resampling by linear interpolation
                n  = len(samples)
                xa = np.arange(round(n*self._ratio))/self._ratio
                xp = np.arange(n)
                samples = np.round(np.interp(xa,xp,samples)).astype(np.int16)

        self._write_samples(samples, {})

    def _process_iq_samples(self, seq, samples, rssi, gps):
        if not self._squelch_status(seq, samples, rssi):
            return

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

        # no GPS or no recent GPS solution
        last = gps['last_gps_solution']
        if last == 255 or last == 254:
            self._options.status = 3

    def _get_output_filename(self):
        if self._options.test_mode:
            return os.devnull
        station = '' if self._options.station is None else '_'+ self._options.station

        # if multiple connections specified but not distinguished via --station then use index
        if self._options.multiple_connections and self._options.station is None:
            station = '_%d' % self._options.idx
        if self._options.filename != '':
            filename = '%s%s.wav' % (self._options.filename, station)
        else:
            ts  = time.strftime('%Y%m%dT%H%M%SZ', self._start_ts)
            filename = '%s_%d%s_%s.wav' % (ts, int(self._freq * 1000), station, self._options.modulation)
        if self._options.dir is not None:
            filename = '%s/%s' % (self._options.dir, filename)
        return filename

    def _update_wav_header(self):
        with open(self._get_output_filename(), 'r+b') as fp:
            fp.seek(0, os.SEEK_END)
            filesize = fp.tell()
            fp.seek(0, os.SEEK_SET)

            # fp.tell() sometimes returns zero. _write_wav_header writes filesize - 8
            if filesize >= 8:
                _write_wav_header(fp, filesize, int(self._output_sample_rate), self._num_channels, self._options.is_kiwi_wav)

    def _write_samples(self, samples, *args):
        """Output to a file on the disk."""
        now = time.gmtime()
        sec_of_day = lambda x: 3600*x.tm_hour + 60*x.tm_min + x.tm_sec
        dt_reached = self._options.dt != 0 and self._start_ts is not None and sec_of_day(now)//self._options.dt != sec_of_day(self._start_ts)//self._options.dt
        if self._start_ts is None or (self._options.filename == '' and dt_reached):
            self._start_ts = now
            self._start_time = time.time()
            # Write a static WAV header
            with open(self._get_output_filename(), 'wb') as fp:
                _write_wav_header(fp, 100, int(self._output_sample_rate), self._num_channels, self._options.is_kiwi_wav)
            if self._options.is_kiwi_tdoa:
                # NB: MUST be a print (i.e. not a logging.info)
                print("file=%d %s" % (self._options.idx, self._get_output_filename()))
            else:
                logging.info("Started a new file: %s" % self._get_output_filename())
        with open(self._get_output_filename(), 'ab') as fp:
            if self._options.is_kiwi_wav:
                gps = args[0]
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

class KiwiWaterfallRecorder(KiwiSDRStream):
    def __init__(self, options):
        super(KiwiWaterfallRecorder, self).__init__()
        self._options = options
        self._type = 'W/F'
        freq = options.frequency
        #logging.info "%s:%s freq=%d" % (options.server_host, options.server_port, freq)
        self._freq = freq
        self._start_ts = None
        self._start_time = None
        self._last_gps = dict(zip(['last_gps_solution', 'dummy', 'gpssec', 'gpsnsec'], [0,0,0,0]))

    def _setup_rx_params(self):
        if self._freq == 1000 and self._options.zoom == 0:
            self._freq = 15000
        self._set_zoom_cf(self._options.zoom, self._freq)
        self._set_maxdb_mindb(-10, -110)    # needed, but values don't matter
        self._set_wf_speed(self._options.speed)
        if self._options.no_api:
            self._setup_no_api()
            return
        #self._set_wf_comp(True)
        self._set_wf_comp(False)
        self._set_wf_interp(self._options.interp)
        self.set_name(self._options.user)
        self._start_time = time.time()
        span = self.zoom_to_span(self._options.zoom)
        start = self._freq - span/2
        stop  = self._freq + span/2
        if start < 0 or stop > self.MAX_FREQ:
            s = "Frequency and zoom values result in span outside 0 - %d kHz range" % (self.MAX_FREQ)
            raise Exception(s)
        logging.info("wf samples: start|center|stop %.1f|%.1f|%.1f kHz, span %d kHz, rbw %.3f kHz, cal %d dB"
              % (start, self._freq, stop, span, span/self.WF_BINS, self._options.cal))

    def _process_waterfall_samples(self, seq, samples):
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
            if s < min and i > 2:   # skip DC offset notch in first two bins
                min = s
                bmin = i
            i += 1
        span = self.zoom_to_span(self._options.zoom)
        start = self._freq - span/2
        logging.info("wf samples: %d bins, min %d dB @ %.1f kHz, max %d dB @ %.1f kHz"
              % (nbins, min-255 + self._options.cal, start + span*bmin/bins, max-255 + self._options.cal, start + span*bmax/bins))

class KiwiExtensionRecorder(KiwiSDRStream):
    def __init__(self, options):
        super(KiwiExtensionRecorder, self).__init__()
        self._options = options
        self._type = 'EXT'
        logging.info("%s:%s EXT init" % (options.server_host, options.server_port))
        self._freq = None
        self._start_ts = None
        self._start_time = None

    def _setup_rx_params(self):
        logging.info("EXT setup_rx_params()")
        self.set_name(self._options.user)
        logging.info("CAUTION: rx_chan=0 assumed!")
        self._send_message('SET ext_switch_to_client=%s first_time=1 rx_chan=0' % (self._options.extension))
        if (self._options.extension == 'DRM'):
            self._send_message('SET lock_set')
        self._start_time = time.time()

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
        opt_single.timestamp = int(time.time() + os.getpid() + i) & 0xffffffff
        for x in ['server_port', 'password', 'tlimit_password', 'frequency', 'agc_gain', 'filename', 'station', 'user']:
            opt_single.__dict__[x] = _sel_entry(i, opt_single.__dict__[x])
        l.append(opt_single)
        multiple_connections = i
    return multiple_connections,l

def get_comma_separated_args(option, opt, value, parser, fn):
    values = [fn(v.strip()) for v in value.split(',')]
    setattr(parser.values, option.dest, values)
##    setattr(parser.values, option.dest, map(fn, value.split(',')))

def join_threads(snd, wf, ext):
    [r._event.set() for r in snd]
    [r._event.set() for r in wf]
    [r._event.set() for r in ext]
    [t.join() for t in threading.enumerate() if t is not threading.currentThread()]

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
                      dest='server_host', type='string',
                      default='localhost', help='Server host (can be a comma-separated list)',
                      action='callback',
                      callback_args=(str,),
                      callback=get_comma_separated_args)
    parser.add_option('-p', '--server-port',
                      dest='server_port', type='string',
                      default=8073, help='Server port, default 8073 (can be a comma-separated list)',
                      action='callback',
                      callback_args=(int,),
                      callback=get_comma_separated_args)
    parser.add_option('--pw', '--password',
                      dest='password', type='string', default='',
                      help='Kiwi login password (if required, can be a comma-separated list)',
                      action='callback',
                      callback_args=(str,),
                      callback=get_comma_separated_args)
    parser.add_option('--tlimit-pw', '--tlimit-password',
                      dest='tlimit_password', type='string', default='',
                      help='Connect time limit exemption password (if required, can be a comma-separated list)',
                      action='callback',
                      callback_args=(str,),
                      callback=get_comma_separated_args)
    parser.add_option('-u', '--user',
                      dest='user', type='string', default='kiwirecorder.py',
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
    parser.add_option('--log', '--log-level', '--log_level', type='choice',
                      dest='log_level', default='warn',
                      choices=['debug', 'info', 'warn', 'error', 'critical'],
                      help='Log level: debug|info|warn(default)|error|critical')
    parser.add_option('-q', '--quiet',
                      dest='quiet',
                      default=False,
                      action='store_true',
                      help='Don\'t print progress messages')
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
    parser.add_option('--connect-retries', '--connect_retries',
                      dest='connect_retries', type='int', default=0,
                      help='Number of retries when connecting to host (retries forever by default)')
    parser.add_option('--connect-timeout', '--connect_timeout',
                      dest='connect_timeout', type='int', default=15,
                      help='Retry timeout(sec) connecting to host')
    parser.add_option('-k', '--socket-timeout', '--socket_timeout',
                      dest='socket_timeout', type='int', default=10,
                      help='Socket timeout(sec) during data transfers')
    parser.add_option('--OV',
                      dest='ADC_OV',
                      default=False,
                      action='store_true',
                      help='Print "ADC OV" message when Kiwi ADC is overloaded')
    parser.add_option('--ts', '--tstamp', '--timestamp',
                      dest='tstamp',
                      default=False,
                      action='store_true',
                      help='Add timestamps to output. Applies only to S-meter mode currently.')
    parser.add_option('--stats',
                      dest='stats',
                      default=False,
                      action='store_true',
                      help='Print additional statistics. Applies only to S-meter mode currently.')
    parser.add_option('-v', '-V', '--version',
                      dest='krec_version',
                      default=False,
                      action='store_true',
                      help='Print version number and exit')

    group = OptionGroup(parser, "Audio connection options", "")
    group.add_option('-f', '--freq',
                      dest='frequency',
                      type='string', default=1000,
                      help='Frequency to tune to, in kHz (can be a comma-separated list). '
                        'For sideband modes (lsb/lsn/usb/usn/cw/cwn) this is the carrier frequency. See --pbc option below.',
                      action='callback',
                      callback_args=(float,),
                      callback=get_comma_separated_args)
    group.add_option('--pbc', '--freq-pbc',
                      dest='freq_pbc',
                      action='store_true', default=False,
                      help='For sideband modes (lsb/lsn/usb/usn/cw/cwn) interpret -f/--freq frequency as the passband center frequency.')
    group.add_option('-m', '--modulation',
                      dest='modulation',
                      type='string', default='am',
                      help='Modulation; one of am/amn, sam/sau/sal/sas/qam, lsb/lsn, usb/usn, cw/cwn, nbfm, iq (default passband if -L/-H not specified)')
    group.add_option('--ncomp', '--no_compression',
                      dest='compression',
                      default=True,
                      action='store_false',
                      help='Don\'t use audio compression')
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
                      type='string',
                      default=None,
                      help='AGC gain; if set, AGC is turned off (can be a comma-separated list)',
                      action='callback',
                      callback_args=(float,),
                      callback=get_comma_separated_args)
    group.add_option('--agc-yaml',
                      dest='agc_yaml_file',
                      type='string',
                      default=None,
                      help='AGC options provided in a YAML-formatted file')
    group.add_option('--scan-yaml',
                      dest='scan_yaml_file',
                      type='string',
                      default=None,
                      help='Scan options provided in a YAML-formatted file')
    group.add_option('--nb',
                      dest='nb',
                      action='store_true', default=False,
                      help='Enable noise blanker with default parameters.')
    group.add_option('--nb-gate',
                      dest='nb_gate',
                      type='int', default=100,
                      help='Noise blanker gate time in usec (100 to 5000, default 100)')
    group.add_option('--nb-th', '--nb-thresh',
                      dest='nb_thresh',
                      type='int', default=50,
                      help='Noise blanker threshold in percent (0 to 100, default 50)')
    group.add_option('--de-emp',
                      dest='de_emp',
                      action='store_true', default=False,
                      help='Enable de-emphasis.')
    group.add_option('-w', '--kiwi-wav',
                      dest='is_kiwi_wav',
                      default=False,
                      action='store_true',
                      help='In the wav file include KIWI header containing GPS time-stamps (only for IQ mode)')
    group.add_option('--kiwi-tdoa',
                      dest='is_kiwi_tdoa',
                      default=False,
                      action='store_true',
                      help='Used when called by Kiwi TDoA extension')
    group.add_option('--test-mode',
                      dest='test_mode',
                      default=False,
                      action='store_true',
                      help='Write wav data to /dev/null (Linux) or NUL (Windows)')
    group.add_option('--snd', '--sound',
                      dest='sound',
                      default=False,
                      action='store_true',
                      help='Also process sound data when in waterfall or S-meter mode (sound connection options above apply)')
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
    group.add_option('--wf',
                      dest='waterfall',
                      default=False,
                      action='store_true',
                      help='Process waterfall data instead of audio')
    group.add_option('-z', '--zoom',
                      dest='zoom', type='int', default=0,
                      help='Zoom level 0-14')
    group.add_option('--speed',
                      dest='speed', type='int', default=1,
                      help='Waterfall update speed: 1=1Hz, 2=slow, 3=med, 4=fast')
    group.add_option('--cal',
                      dest='cal', type='int', default=0,
                      help='Waterfall calibration correction value.')
    group.add_option('--interp',
                      dest='interp', type='int', default=13,
                      help='Waterfall display interpolation 0-13')
    parser.add_option_group(group)

    group = OptionGroup(parser, "KiwiSDR development options", "")
    group.add_option('--gc-stats',
                      dest='gc_stats',
                      default=False,
                      action='store_true',
                      help='Print garbage collection stats')
    group.add_option('--ext',
                      dest='extension',
                      default=None,
                      type='string',
                      help='Also open a connection to EXTENSION name')
    group.add_option('--nolocal',
                      dest='nolocal',
                      default=False,
                      action='store_true',
                      help='Make local network connections appear non-local')
    group.add_option('--no-api',
                      dest='no_api',
                      default=False,
                      action='store_true',
                      help='Simulate connection to Kiwi using improper/incomplete API')
    parser.add_option_group(group)

    opts_no_defaults = optparse.Values()
    __, args = parser.parse_args(values=opts_no_defaults)
    options = optparse.Values(parser.get_default_values().__dict__)
    options._update_careful(opts_no_defaults.__dict__)

    ## clean up OptionParser which has cyclic references
    parser.destroy()

    if options.krec_version:
        print('kiwirecorder v1.0')
        sys.exit()

    FORMAT = '%(asctime)-15s pid %(process)5d %(message)s'
    logging.basicConfig(level=logging.getLevelName(options.log_level.upper()), format=FORMAT)
    if options.gc_stats:
        gc.set_debug(gc.DEBUG_SAVEALL | gc.DEBUG_LEAK | gc.DEBUG_UNCOLLECTABLE)

    run_event = threading.Event()
    run_event.set()

    if options.S_meter >= 0:
        if options.S_meter > 0 and options.sdt != 0:
            raise Exception('Options --S-meter > 0 and --sdt-sec != 0 are incompatible. Did you mean to use --S-meter=0 ?')
        options.quiet = True

    if options.tlimit is not None and options.dt != 0:
        print('Warning: --tlimit ignored when --dt-sec option used')

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

    ### decode AGC YAML file options
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
                options.scan_state = 'WAIT'
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

    options.raw = False
    options.rigctl_enabled = False
    gopt = options
    multiple_connections,options = options_cross_product(options)

    snd_recorders = []
    if not gopt.waterfall or (gopt.waterfall and gopt.sound):
        for i,opt in enumerate(options):
            opt.multiple_connections = multiple_connections
            opt.idx = i
            snd_recorders.append(KiwiWorker(args=(KiwiSoundRecorder(opt),opt,run_event)))

    wf_recorders = []
    if gopt.waterfall:
        for i,opt in enumerate(options):
            opt.multiple_connections = multiple_connections
            opt.idx = i
            wf_recorders.append(KiwiWorker(args=(KiwiWaterfallRecorder(opt),opt,run_event)))

    ext_recorders = []
    if gopt.extension is not None:
        for i,opt in enumerate(options):
            opt.multiple_connections = multiple_connections
            opt.idx = i
            ext_recorders.append(KiwiWorker(args=(KiwiExtensionRecorder(opt),opt,run_event)))

    try:
        for i,r in enumerate(snd_recorders):
            if opt.launch_delay != 0 and i != 0 and options[i-1].server_host == options[i].server_host:
                time.sleep(opt.launch_delay)
            r.start()
            #logging.info("started sound recorder %d, timestamp=%d" % (i, options[i].timestamp))
            logging.info("started sound recorder %d" % i)

        for i,r in enumerate(wf_recorders):
            if i!=0 and options[i-1].server_host == options[i].server_host:
                time.sleep(opt.launch_delay)
            r.start()
            logging.info("started waterfall recorder %d" % i)

        for i,r in enumerate(ext_recorders):
            if i!=0 and options[i-1].server_host == options[i].server_host:
                time.sleep(opt.launch_delay)
            r.start()
            logging.info("started extension recorder %d" % i)

        while run_event.is_set():
            time.sleep(.1)

    except KeyboardInterrupt:
        run_event.clear()
        join_threads(snd_recorders, wf_recorders, ext_recorders)
        print("KeyboardInterrupt: threads successfully closed")
    except Exception as e:
        print_exc()
        run_event.clear()
        join_threads(snd_recorders, wf_recorders, ext_recorders)
        print("Exception: threads successfully closed")

    if gopt.is_kiwi_tdoa:
      for i,opt in enumerate(options):
          # NB: MUST be a print (i.e. not a logging.info)
          print("status=%d,%d" % (i, opt.status))

    if gopt.gc_stats:
        logging.debug('gc %s' % gc.garbage)

if __name__ == '__main__':
    #import faulthandler
    #faulthandler.enable()
    main()
# EOF
