#!/usr/bin/env python
## -*- python -*-

##
## FIXME: should support the usual suspects:
##  IQ-swap, endian-reversal, resampling? (e.g. see kiwirecorder),
##  option to include GPS data, ...
##

import array, logging, os, struct, sys, time, copy, threading, os
import gc
import numpy as np

from copy import copy
from traceback import print_exc
from kiwi import KiwiSDRStream, KiwiWorker
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

class RingBuffer(object):
    def __init__(self, len):
        self._array = np.zeros(65, dtype='float')
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

    def median(self):
        return np.median(self._array)


class Squelch(object):
    def __init__(self, options):
        self._status_msg  = False
        self._threshold   = options.thresh
        self._tail_delay  = round(options.squelch_tail*12000/512) ## seconds to number of buffers
        self._ring_buffer = RingBuffer(65)
        self._squelch_on_seq = None

    def process(self, seq, rssi):
        if not self._ring_buffer.is_filled() or self._squelch_on_seq is None:
            self._ring_buffer.insert(rssi)
        if not self._ring_buffer.is_filled():
            return False
        median_nf   = self._ring_buffer.median()
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

class KiwiNetcat(KiwiSDRStream):
    def __init__(self, options, reader):
        super(KiwiNetcat, self).__init__()
        self._options = options
        self._type = 'admin' if options.admin is True else ('W/F' if options.waterfall is True else 'SND')
        self._reader = reader
        freq = options.frequency
        #logging.info("%s:%s freq=%d" % (options.server_host, options.server_port, freq))
        self._freq = freq
        self._start_ts = None
        #self._start_time = None
        self._start_time = time.time()
        self._options.stats = None
        self._squelch = Squelch(self._options) if options.thresh is not None else None
        self._last_gps = dict(zip(['last_gps_solution', 'dummy', 'gpssec', 'gpsnsec'], [0,0,0,0]))
        self._fp_stdout = os.fdopen(sys.stdout.fileno(), 'wb')

    def _setup_rx_params(self):
        self.set_name(self._options.user)

        if self._type == 'SND':
            mod    = self._options.modulation
            lp_cut = self._options.lp_cut
            hp_cut = self._options.hp_cut
            if mod == 'am' or mod == 'amn':
                # For AM, ignore the low pass filter cutoff
                lp_cut = -hp_cut
            if mod == 'lsb':
                # For LSB, transpose and invert the cutoffs
                lp_cut2 = -hp_cut
                hp_cut2 = -lp_cut
                lp_cut = lp_cut2
                hp_cut = hp_cut2
            self.set_mod(mod, lp_cut, hp_cut, self._freq)

            if self._options.agc_gain != None: ## fixed gain (no AGC)
                self.set_agc(on=False, gain=self._options.agc_gain)
            if self._options.agc_decay != None: ## AGC on with specified decay
                self.set_agc(on=True, decay=self._options.agc_decay)
            elif self._options.agc_yaml_file != None: ## custon AGC parameters from YAML file
                self.set_agc(**self._options.agc_yaml)
            else: ## default is AGC ON (with default parameters)
                self.set_agc(on=True)

            if self._options.compression is False:
                self._set_snd_comp(False)

            if self._options.de_emp is True:
                self.set_de_emp(1)

        else:   # waterfall
            self._set_maxdb_mindb(-10, -110)    # needed, but values don't matter
            self._set_zoom_cf(0, 0)
            self._set_wf_comp(False)
            self._set_wf_speed(1)   # 1 Hz update

    def _process_audio_samples_raw(self, seq, samples, rssi):
        if self._options.progress is True:
            sys.stdout.write('\rBlock: %08x, RSSI: %6.1f' % (seq, rssi))
            sys.stdout.flush()
        else:
            if self._squelch:
                is_open = self._squelch.process(seq, rssi)
                if not is_open:
                    self._start_ts = None
                    self._start_time = None
                    return
            self._write_samples(samples, {})

    def _process_iq_samples_raw(self, seq, data):
        count = len(data) // 2
        samples = np.ndarray(count, dtype='>h', buffer=data).astype(np.int16)
        self._write_samples(samples, {})

    def _process_waterfall_samples_raw(self, samples, seq):
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
            sys.stdout.write('\rwf samples %d bins %d..%d dB %.1f..%.1f kHz rbw %d kHz'
                  % (nbins, min-255, max-255, span*bmin/bins, span*bmax/bins, span/bins))
            sys.stdout.flush()
        else:
            self._fp_stdout.write(samples)
            self._fp_stdout.flush()

    def _write_samples(self, samples, *args):
        if self._options.progress is False:
            self._fp_stdout.write(samples)
            self._fp_stdout.flush()

    def _writer_message(self):
        if self._options.writer_init == False:
            self._options.writer_init = True
            if self._options.admin:
                return 'ADM tunO'
        msg = sys.stdin.readline()  # blocks
        return ('ADM tunW %s' % msg) if self._options.admin else msg

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
        for x in ['server_port', 'password', 'tlimit_password', 'frequency', 'agc_gain', 'user']:
            opt_single.__dict__[x] = _sel_entry(i, opt_single.__dict__[x])
        l.append(opt_single)
        multiple_connections = i
    return multiple_connections,l

def get_comma_separated_args(option, opt, value, parser, fn):
    values = [fn(v.strip()) for v in value.split(',')]
    setattr(parser.values, option.dest, values)
##    setattr(parser.values, option.dest, map(fn, value.split(',')))

def join_threads(nc):
    [r._event.set() for r in nc]
    [t.join() for t in threading.enumerate() if t is not threading.currentThread()]

def main():
    parser = OptionParser()
    parser.add_option('--log', '--log-level', '--log_level', type='choice',
                      dest='log_level', default='warn',
                      choices=['debug', 'info', 'warn', 'error', 'critical'],
                      help='Log level: debug|info|warn(default)|error|critical')
    parser.add_option('--progress',
                      dest='progress',
                      default=False,
                      action='store_true',
                      help='Print progress messages instead of output of binary data')
    parser.add_option('-k', '--socket-timeout', '--socket_timeout',
                      dest='socket_timeout', type='int', default=10,
                      help='Timeout(sec) for sockets')
    parser.add_option('-s', '--server-host',
                      dest='server_host', type='string',
                      default='localhost', help='Server host (can be a comma-delimited list)',
                      action='callback',
                      callback_args=(str,),
                      callback=get_comma_separated_args)
    parser.add_option('-p', '--server-port',
                      dest='server_port', type='string',
                      default=8073, help='Server port, default 8073 (can be a comma delimited list)',
                      action='callback',
                      callback_args=(int,),
                      callback=get_comma_separated_args)
    parser.add_option('--pw', '--password',
                      dest='password', type='string', default='',
                      help='Kiwi login password (if required, can be a comma delimited list)',
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
                      dest='user', type='string', default='kiwi_nc.py',
                      help='Kiwi connection user name',
                      action='callback',
                      callback_args=(str,),
                      callback=get_comma_separated_args)
    parser.add_option('--launch-delay', '--launch_delay',
                      dest='launch_delay',
                      type='int', default=0,
                      help='Delay (secs) in launching multiple connections')
    parser.add_option('-f', '--freq',
                      dest='frequency',
                      type='string', default=1000,
                      help='Frequency to tune to, in kHz (can be a comma-separated list). '
                        'For sideband modes (lsb/lsn/usb/usn/cw/cwn) this is the carrier frequency. See --pbc option below.',
                      action='callback',
                      callback_args=(float,),
                      callback=get_comma_separated_args)
    parser.add_option('--pbc', '--freq-pbc',
                      dest='freq_pbc',
                      action='store_true', default=False,
                      help='For sideband modes (lsb/lsn/usb/usn/cw/cwn) interpret -f/--freq frequency as the passband center frequency.')
    parser.add_option('-m', '--modulation',
                      dest='modulation',
                      type='string', default='am',
                      help='Modulation; one of am/amn, sam/sau/sal/sas/qam, lsb/lsn, usb/usn, cw/cwn, nbfm, iq (default passband if -L/-H not specified)')
    parser.add_option('--ncomp', '--no_compression',
                      dest='compression',
                      default=True,
                      action='store_false',
                      help='Don\'t use audio compression (IQ mode never uses compression)')
    parser.add_option('-L', '--lp-cutoff',
                      dest='lp_cut',
                      type='float', default=100,
                      help='Low-pass cutoff frequency, in Hz')
    parser.add_option('-H', '--hp-cutoff',
                      dest='hp_cut',
                      type='float', default=2600,
                      help='Low-pass cutoff frequency, in Hz')
    parser.add_option('--tlimit', '--time-limit',
                      dest='tlimit',
                      type='float', default=None,
                      help='Record time limit in seconds')
    parser.add_option('-T', '--squelch-threshold',
                      dest='thresh',
                      type='float', default=None,
                      help='Squelch threshold, in dB.')
    parser.add_option('--squelch-tail',
                      dest='squelch_tail',
                      type='float', default=1,
                      help='Time for which the squelch remains open after the signal is below threshold.')
    parser.add_option('-g', '--agc-gain',
                      dest='agc_gain',
                      type='string',
                      default=None,
                      help='AGC gain; if set, AGC is turned off (can be a comma-separated list)',
                      action='callback',
                      callback_args=(float,),
                      callback=get_comma_separated_args)
    parser.add_option('--agc-decay',
                      dest='agc_decay',
                      type='int',
                      default=1000,
                      help='AGC decay (msec); if set, AGC is turned on')
    parser.add_option('--agc-yaml',
                      dest='agc_yaml_file',
                      type='string',
                      default=None,
                      help='AGC options provided in a YAML-formatted file')
    parser.add_option('--de-emp',
                      dest='de_emp',
                      action='store_true', default=False,
                      help='Enable de-emphasis.')
    parser.add_option('--wf', '--waterfall',
                      dest='waterfall',
                      default=False,
                      action='store_true',
                      help='Process waterfall data instead of audio')

    group = OptionGroup(parser, "KiwiSDR development options", "")
    group.add_option('--gc-stats',
                      dest='gc_stats',
                      default=False,
                      action='store_true',
                      help='Print garbage collection stats')
    group.add_option('--admin',
                      dest='admin',
                      default=False, action='store_true',
                      help='Kiwi connection: admin instead of default audio stream.')
    parser.add_option_group(group)

    (options, unused_args) = parser.parse_args()

    ## clean up OptionParser which has cyclic references
    parser.destroy()

    FORMAT = '%(asctime)-15s pid %(process)5d %(message)s'
    logging.basicConfig(level=logging.getLevelName(options.log_level.upper()), format=FORMAT)
    if options.gc_stats:
        gc.set_debug(gc.DEBUG_SAVEALL | gc.DEBUG_LEAK | gc.DEBUG_UNCOLLECTABLE)

    run_event = threading.Event()
    run_event.set()

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

    options.raw = True
    options.S_meter = -1
    options.ADC_OV = None
    options.is_kiwi_tdoa = False
    options.no_api = False
    options.nolocal = False
    options.connect_retries = 0
    options.connect_timeout = 3
    options.rigctl_enabled = False
    gopt = options
    multiple_connections,options = options_cross_product(options)

    nc_inst = []
    for i,opt in enumerate(options):
        opt.multiple_connections = multiple_connections
        opt.idx = 0
        nc_inst.append(KiwiWorker(args=(KiwiNetcat(opt, True),opt,run_event)))
        if gopt.admin:
            opt.writer_init = False
            opt.idx = 1
            nc_inst.append(KiwiWorker(args=(KiwiNetcat(opt, False),opt,run_event)))

    try:
        for i,r in enumerate(nc_inst):
            if opt.launch_delay != 0 and i != 0 and options[i-1].server_host == options[i].server_host:
                time.sleep(opt.launch_delay)
            r.start()
            #logging.info("started sound recorder %d, timestamp=%d" % (i, options[i].timestamp))
            logging.info("started sound recorder %d" % i)

        while run_event.is_set():
            time.sleep(.1)

    except KeyboardInterrupt:
        run_event.clear()
        join_threads(nc_inst)
        print("KeyboardInterrupt: threads successfully closed")
    except Exception as e:
        print_exc()
        run_event.clear()
        join_threads(nc_inst)
        print("Exception: threads successfully closed")

    if gopt.gc_stats:
        logging.debug('gc %s' % gc.garbage)

if __name__ == '__main__':
    #import faulthandler
    #faulthandler.enable()
    main()
# EOF
