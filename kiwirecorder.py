## -*- python -*-

import array, codecs, logging, os, struct, sys, time, traceback, copy, threading
from optparse import OptionParser
import numpy as np


import kiwiclient

def _write_wav_header(fp, filesize, samplerate, num_channels, is_kiwi_wav):
    fp.write(struct.pack('<4sI4s', 'RIFF', filesize - 8, 'WAVE'))
    bits_per_sample = 16
    byte_rate       = samplerate * num_channels * bits_per_sample / 8
    block_align     = num_channels * bits_per_sample / 8
    fp.write(struct.pack('<4sIHHIIHH', 'fmt ', 16, 1, num_channels, samplerate, byte_rate, block_align, bits_per_sample))
    if not is_kiwi_wav:
        fp.write(struct.pack('<4sI', 'data', filesize - 12 - 8 - 16 - 8))

class KiwiSoundRecorder(kiwiclient.KiwiSDRStream):
    def __init__(self, options):
        super(KiwiSoundRecorder, self).__init__()
        self._options = options
        self._isWF = False
        freq = options.frequency
        #print "%s:%s freq=%d" % (options.server_host, options.server_port, freq)
        self._freq = freq
        self._start_ts = None
        self._squelch_on_seq = None
        self._nf_array = array.array('i')
        for x in xrange(65):
            self._nf_array.insert(x, 0)
        self._nf_samples = 0
        self._nf_index = 0
        self._num_channels = 2 if options.modulation == 'iq' else 1
        self._last_gps = dict(zip(['last_gps_solution', 'dummy', 'gpssec', 'gpsnsec'], [0,0,0,0]))

    def _setup_rx_params(self):
        mod    = self._options.modulation
        lp_cut = self._options.lp_cut
        hp_cut = self._options.hp_cut
        if mod == 'am':
            # For AM, ignore the low pass filter cutoff
            lp_cut = -hp_cut
        self.set_mod(mod, lp_cut, hp_cut, self._freq)
        if self._options.agc_gain != None:
            self.set_agc(on=False, gain=self._options.agc_gain)
        else:
            self.set_agc(on=True)
        self.set_inactivity_timeout(0)
        self.set_name('kiwirecorder.py')

    def _process_audio_samples(self, seq, samples, rssi):
        sys.stdout.write('\rBlock: %08x, RSSI: %-04d' % (seq, rssi))
        if self._nf_samples < len(self._nf_array) or self._squelch_on_seq is None:
            self._nf_array[self._nf_index] = rssi
            self._nf_index += 1
            if self._nf_index == len(self._nf_array):
                self._nf_index = 0
        if self._nf_samples < len(self._nf_array):
            self._nf_samples += 1
            return

        median_nf = sorted(self._nf_array)[len(self._nf_array) // 3]
        rssi_thresh = median_nf + self._options.thresh
        is_open = self._squelch_on_seq is not None
        if is_open:
            rssi_thresh -= 6
        rssi_green = rssi >= rssi_thresh
        if rssi_green:
            self._squelch_on_seq = seq
            is_open = True
        sys.stdout.write(' Median: %-04d Thr: %-04d %s' % (median_nf, rssi_thresh, ("s", "S")[is_open]))
        if not is_open:
            return
        if seq > self._squelch_on_seq + 45:
            print "\nSquelch closed"
            self._squelch_on_seq = None
            self._start_ts = None
            return
        self._write_samples(samples, {})

    def _process_iq_samples(self, seq, samples, rssi, gps):
        ##print gps['gpsnsec']-self._last_gps['gpsnsec']
        self._last_gps = gps
        ## convert list of complex numbers into an array
        s = array.array('h')
        for x in [[y.real, y.imag] for y in samples]:
            s.extend(map(int, x))
        self._write_samples(s, gps)

    def _get_output_filename(self):
        ts  = time.strftime('%Y%m%dT%H%M%SZ', self._start_ts)
        sta = '' if self._options.station is "None" else '_'+ self._options.station
        return '%s_%d%s_%s.wav' % (ts, int(self._freq * 1000), sta, self._options.modulation)

    def _update_wav_header(self):
        with open(self._get_output_filename(), 'r+b') as fp:
            fp.seek(0, os.SEEK_END)
            filesize = fp.tell()
            fp.seek(0, os.SEEK_SET)
            _write_wav_header(fp, filesize, int(self._sample_rate), self._num_channels, self._options.is_kiwi_wav)

    def _write_samples(self, samples, *args):
        """Output to a file on the disk."""
        # print '_write_samples', args
        now = time.gmtime()
        if self._start_ts is None or self._start_ts.tm_hour != now.tm_hour:
            self._start_ts = now
            # Write a static WAV header
            with open(self._get_output_filename(), 'wb') as fp:
                _write_wav_header(fp, 100, int(self._sample_rate), self._num_channels, self._options.is_kiwi_wav)
            print "\nStarted a new file: %s" % (self._get_output_filename())
        with open(self._get_output_filename(), 'ab') as fp:
            if self._options.is_kiwi_wav:
                gps = args[0]
                logging.info('%s: last_gps_solution=%d gpssec=(%d,%d)' % (self._get_output_filename(), gps['last_gps_solution'], gps['gpssec'], gps['gpsnsec']));
                fp.write(struct.pack('<4sIBBII', 'kiwi', 10, gps['last_gps_solution'], 0, gps['gpssec'], gps['gpsnsec']))
                sample_size = samples.itemsize * len(samples)
                fp.write(struct.pack('<4sI', 'data', sample_size))
            # TODO: something better than that
            samples.tofile(fp)
        self._update_wav_header()


class KiwiWaterfallRecorder(kiwiclient.KiwiSDRStream):
    def __init__(self, options):
        super(KiwiWaterfallRecorder, self).__init__()
        self._options = options
        self._isWF = True
        freq = options.frequency
        #print "%s:%s freq=%d" % (options.server_host, options.server_port, freq)
        self._freq = freq
        self._start_ts = None

        # xxx
        self._squelch_on_seq = None
        self._nf_array = array.array('i')
        for x in xrange(65):
            self._nf_array.insert(x, 0)
        self._nf_samples = 0
        self._nf_index = 0
        self._num_channels = 2 if options.modulation == 'iq' else 1
        self._last_gps = dict(zip(['last_gps_solution', 'dummy', 'gpssec', 'gpsnsec'], [0,0,0,0]))

        self.f_wf = open('waterfall.txt', 'wb')


    def _setup_rx_params(self):
        self._set_zoom_start(0, 0)
        self._set_maxdb_mindb(-10, -110)    # needed, but values don't matter
        #self._set_wf_comp(True)
        self._set_wf_comp(False)
        self._set_wf_speed(4)   # 1 Hz update
        self.set_inactivity_timeout(0)
        self.set_name('kiwirecorder.py')

    def _process_waterfall_samples(self, seq, samples):
        nbins = len(samples)
        bins = nbins-1

        i_max = np.argmax(samples)
        v_max = samples[i_max]
        i_min = np.argmin(samples)
        v_min = samples[i_min]

        span = 30000
        bin_samples = bytearray(list(np.array(samples).astype(np.uint8)))
        
        self.f_wf.write(bin_samples)
        print nbins, "wf samples %d bins %d..%d dB %.1f..%.1f kHz rbw %d kHz" % (nbins, v_min-255, v_max-255, span*i_min/bins, span*i_max/bins, span/bins)

    def _get_output_filename(self):
        ts  = time.strftime('%Y%m%dT%H%M%SZ', self._start_ts)
        sta = '' if self._options.station is "None" else '_'+ self._options.station
        return '%s_%d%s_%s.wav' % (ts, int(self._freq * 1000), sta, self._options.modulation)

    def _update_wav_header(self):
        with open(self._get_output_filename(), 'r+b') as fp:
            fp.seek(0, os.SEEK_END)
            filesize = fp.tell()
            fp.seek(0, os.SEEK_SET)
            _write_wav_header(fp, filesize, int(self._sample_rate), self._num_channels, self._options.is_kiwi_wav)

    def _write_samples(self, samples, *args):
        """Output to a file on the disk."""
        # print '_write_samples', args
        now = time.gmtime()
        if self._start_ts is None or self._start_ts.tm_hour != now.tm_hour:
            self._start_ts = now
            # Write a static WAV header
            with open(self._get_output_filename(), 'wb') as fp:
                _write_wav_header(fp, 100, int(self._sample_rate), self._num_channels, self._options.is_kiwi_wav)
            print "\nStarted a new file: %s" % (self._get_output_filename())
        with open(self._get_output_filename(), 'ab') as fp:
            if self._options.is_kiwi_wav:
                gps = args[0]
                logging.info('%s: last_gps_solution=%d gpssec=(%d,%d)' % (self._get_output_filename(), gps['last_gps_solution'], gps['gpssec'], gps['gpsnsec']));
                fp.write(struct.pack('<4sIBBII', 'kiwi', 10, gps['last_gps_solution'], 0, gps['gpssec'], gps['gpsnsec']))
                sample_size = samples.itemsize * len(samples)
                fp.write(struct.pack('<4sI', 'data', sample_size))
            # TODO: something better than that
            samples.tofile(fp)
        self._update_wav_header()

    def close(self):
        try:
            self._stream.close_connection()
            self._socket.close()
            self.f_wf.close()
        except Exception as e:
            print "exception: %s" % e


class Worker(threading.Thread):
   def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
       super(Worker, self).__init__(group=group, target=target, name=name, verbose=verbose)
       self._recorder, self._options, self._run_event = args

   def _do_run(self):
       return self._run_event.is_set()

   def _sleep(self, seconds):
       for i in range(seconds):
           if not self._do_run():
               break;
           time.sleep(1)

   def run(self):
       while self._do_run():
               try:
                   self._recorder.connect(self._options.server_host, self._options.server_port)
               except:
                   print "Failed to connect, sleeping and reconnecting"
                   self._sleep(15)
                   continue

               try:
                   self._recorder.open()
                   while self._do_run():
                       self._recorder.run()
               except kiwiclient.KiwiTooBusyError:
                   print "Server %s:%d too busy now" % (self._options.server_host, self._options.server_port)
                   self._sleep(15)
                   continue
               except Exception as e:
                   traceback.print_exc()
                   break

       self._recorder.close()
       print "exiting"


def options_cross_product(options):
    """build a list of options according to the number of servers specified"""
    def _sel_entry(i, l):
        """if l is a list, return the element with index i, else return l"""
        return l[min(i, len(l)-1)] if type(l) == list else l

    l = []
    for i,s in enumerate(options.server_host):
        opt_single = copy.copy(options)
        opt_single.server_host = s;
        for x in ['server_port', 'frequency', 'agc_gain', 'station']:
            opt_single.__dict__[x] = _sel_entry(i, opt_single.__dict__[x])
        l.append(opt_single)
    return l

def get_comma_separated_args(option, opt, value, parser, fn):
    setattr(parser.values, option.dest, map(fn, value.split(',')))

def main():
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout)

    parser = OptionParser()
    parser.add_option('--log-level', '--log_level', type='choice',
                      dest='log_level', default='warn',
                      choices=['debug', 'info', 'warn', 'error', 'critical'],
                      help='Log level: debug|info|warn|error|critical')
    parser.add_option('-k', '--socket-timeout', '--socket_timeout',
                      dest='socket_timeout', type='int', default=10,
                      help='Timeout(sec) for sockets')
    parser.add_option('-s', '--server-host',
                      dest='server_host', type='string',
                      default='localhost', help='server host (can be a comma-delimited list)',
                      action='callback',
                      callback_args=(str,),
                      callback=get_comma_separated_args)
    parser.add_option('-p', '--server-port',
                      dest='server_port', type='string',
                      default=8073, help='server port, default 8073 (can be a comma delimited list)',
                      action='callback',
                      callback_args=(int,),
                      callback=get_comma_separated_args)
    parser.add_option('-f', '--freq',
                      dest='frequency',
                      type='string', default=1000,
                      help='Frequency to tune to, in kHz (can be a comma-separated list)',
                      action='callback',
                      callback_args=(float,),
                      callback=get_comma_separated_args)
    parser.add_option('-m', '--modulation',
                      dest='modulation',
                      type='string', default='am',
                      help='Modulation; one of am, lsb, usb, cw, nbfm, iq')
    parser.add_option('-L', '--lp-cutoff',
                      dest='lp_cut',
                      type='float', default=100,
                      help='Low-pass cutoff frequency, in Hz.')
    parser.add_option('-H', '--hp-cutoff',
                      dest='hp_cut',
                      type='float', default=2600,
                      help='Low-pass cutoff frequency, in Hz.')
    parser.add_option('--station',
                      dest='station',
                      type='string', default="None",
                      help='Station ID to be appended (can be a comma-separated list)',
                      action='callback',
                      callback_args=(str,),
                      callback=get_comma_separated_args)
    parser.add_option('-T', '--threshold',
                      dest='thresh',
                      type='float', default=0,
                      help='Squelch threshold, in dB.')
    parser.add_option('-w', '--kiwi-wav',
                      dest='is_kiwi_wav',
                      default=False,
                      action='store_true',
                      help='wav file format including KIWI header (GPS time-stamps) only for IQ mode')
    parser.add_option('-g', '--agc-gain',
                      dest='agc_gain',
                      type='string',
                      default=None,
                      help='AGC gain; if set, AGC is turned off (can be a comma-separated list)',
                      action='callback',
                      callback_args=(float,),
                      callback=get_comma_separated_args)
    parser.add_option('-z', '--zoom',
                      dest='zoom', type='int', default=0,
                      help='Zoom level 0-14')
    parser.add_option('--wf',
                      dest='waterfall',
                      default=False,
                      action='store_true',
                      help='Process waterfall data instead of audio')
    parser.add_option('--snd',
                      dest='sound',
                      default=False,
                      action='store_true',
                      help='Also process sound data when in waterfall mode')

    (options, unused_args) = parser.parse_args()

    logging.basicConfig(level=logging.getLevelName(options.log_level.upper()))

    run_event = threading.Event()
    run_event.set()

    gopt = options
    options = options_cross_product(options)

    snd_recorders = []
    if not gopt.waterfall or (gopt.waterfall and gopt.sound):
        snd_recorders = [Worker(args=(KiwiSoundRecorder(opt),opt,run_event)) for i,opt in enumerate(options)]

    wf_recorders = []
    if gopt.waterfall:
        for i,opt in enumerate(options):
            wf_recorders.append(Worker(args=(KiwiWaterfallRecorder(opt),opt,run_event)))

    try:
        for i,r in enumerate(snd_recorders):
            if i!=0 and options[i-1].server_host == options[i].server_host:
                time.sleep(2)
            r.start()
            print "started sound recorder %d" % i

        for i,r in enumerate(wf_recorders):
            if i!=0 and options[i-1].server_host == options[i].server_host:
                time.sleep(2)
            r.start()
            print "started waterfall recorder %d" % i

        while True:
            time.sleep(.1)
    except KeyboardInterrupt:
        run_event.clear()
        [t.join() for t in threading.enumerate() if t is not threading.currentThread()]
        print "threads successfully closed"

if __name__ == '__main__':
    main()

# EOF
