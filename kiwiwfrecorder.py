#!/usr/bin/env python3
## -*- python -*-

## to be merged into kiwirecorder.py

import gc, logging, os, time, threading, os
import numpy as np
from traceback import print_exc
from kiwi import KiwiSDRStream, KiwiWorker
from optparse import OptionParser
try:
    from Queue import Queue,Empty  ## python2
except ImportError:
    from queue import Queue,Empty  ## python3


class KiwiSoundRecorder(KiwiSDRStream):
    def __init__(self, options, q):
        super(KiwiSoundRecorder, self).__init__()
        self._options = options
        self._queue = q
        self._type = 'SND'
        self._freq = options.frequency
        self._num_skip = 2 ## skip data at the start of the WS stream with seq < 2

    def _setup_rx_params(self):
        self.set_name(self._options.user)
        mod    = 'iq'
        lp_cut = -1000
        hp_cut = +1000
        self.set_mod(mod, lp_cut, hp_cut, self._freq)
        self.set_agc(on=True)

    def _process_iq_samples(self, seq, samples, rssi, gps):
        if self._num_skip != 0:
            if seq < 2:
                print('IQ: skipping seq=', seq)
                self._num_skip -= 1
                return
            else:
                self._num_skip = 0
        gps_time = gps['gpssec'] + 1e-9*gps['gpsnsec']
        self._queue.put([seq, gps_time])

class KiwiWaterfallRecorder(KiwiSDRStream):
    def __init__(self, options, q):
        super(KiwiWaterfallRecorder, self).__init__()
        self._options = options
        self._queue = q
        self._type = 'W/F'
        self._freq = options.frequency
        self._zoom = options.zoom
        self._freq_bins = None
        self._num_channels = 2
        self._num_skip = 2 ## skip data at the start of the WS stream with seq < 2

    def _setup_rx_params(self):
        self._set_zoom_cf(self._zoom, self._freq)
        self._set_maxdb_mindb(-10, -110)    # needed, but values don't matter
        self._freq_bins = self._freq + (0.5+np.arange(self.WF_BINS))/self.WF_BINS * self.zoom_to_span(self._options.zoom)
        #self._set_wf_comp(True)
        self._set_wf_comp(False)
        self._set_wf_speed(1)   # 1 Hz update
        self.set_name(self._options.user)

    def _process_waterfall_samples(self, seq, samples):
        if self._num_skip != 0:
            if seq < 2:
                self._num_skip -= 1
                return
            else:
                self._num_skip = 0
        logging.info('process_wf_samples: seq= %5d %s' % (seq, samples))
        self._queue.put({'seq':        seq,
                         'freq_bins':  self._freq_bins,
                         'wf_samples': samples})

class Consumer(threading.Thread):
    """ Combines WF data with precise GNSS timestamps from the SND stream """
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None):
        super(Consumer, self).__init__(group=group, target=target, name=name)
        self._options, self._snd_queue, self._wf_queue, self._run_event = args
        self._event     = threading.Event()
        self._store    = dict()
        self._wf_data  = None
        self._start_ts = None

    def run(self):
        while self._run_event.is_set():
            try:
                self.proc()
            except Exception:
                print_exc()
                break
        self._run_event.clear()   # tell all other threads to stop

    def proc(self):
        if self._wf_data is None:
            try:
                self._wf_data = self._wf_queue.get(timeout=1)
            except Empty:
                return

        self.update_store()
        if self._wf_data['seq'] in self._store:
            now = time.gmtime()
            if self._start_ts is None:
                self._start_ts = now
                with open(self._get_output_filename(), 'wb') as f:
                    np.save(f, self._wf_data['freq_bins'])

            ## GNSS timestamp for seq obtained from the SND WS stream
            ts = self._store.pop(self._wf_data['seq'])
            logging.info('found seq %d %f %d (%d|%d,%d)'
                         % (self._wf_data['seq'], ts, len(self._wf_data['wf_samples']),
                            len(self._store), self._wf_queue.qsize(), self._snd_queue.qsize()))
            with open(self._get_output_filename(), 'ab') as f:
                np.save(f, np.array((ts, self._wf_data['wf_samples']),
                                    dtype=[('ts', np.float64), ('wf', ('B', 1024))]))

            self.prune_store(ts)
            self._wf_data = None
        else:
            time.sleep(0.1)

    def _get_output_filename(self):
        station = '' if self._options.station is None else '_'+ self._options.station
        return '%s_%d-%d%s.npy' % (time.strftime('%Y%m%dT%H%M%SZ', self._start_ts),
                                   round(self._wf_data['freq_bins'][ 0] * 1000),
                                   round(self._wf_data['freq_bins'][-1] * 1000),
                                   station)

    def update_store(self):
        """ put all available timestamps from the SND stream into the store """
        while True:
            try:
                seq,ts = self._snd_queue.get(timeout=0.01)
                self._store[seq] = ts
            except Empty:
                break
            except Exception:
                print_exc()
                break

    def prune_store(self, ts):
        """ remove all timestamps before 'ts' """
        for x in list(self._store.items()):
            k,v = x
            if v < ts:
                self._store.pop(k)

def join_threads(threads):
    [t._event.set() for t in threads]
    [t.join()       for t in threading.enumerate() if t is not threading.current_thread()]

def main():
    parser = OptionParser()
    parser.add_option('-s', '--server-host',
                      dest='server_host', type='string',
                      default='localhost', help='Server host')
    parser.add_option('-p', '--server-port',
                      dest='server_port', type='string',
                      default="8073", help='Server port, default 8073')
    parser.add_option('--pw', '--password',
                      dest='password', type='string', default='',
                      help='Kiwi login password')
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
                      dest='socket_timeout', type='int', default=10,
                      help='Timeout(sec) for sockets')
    parser.add_option('--tlimit-pw', '--tlimit-password',
                      dest='tlimit_password', type='string', default='',
                      help='Connect time limit exemption password (if required)')
    parser.add_option('-u', '--user',
                      dest='user', type='string', default='kiwirecorder.py',
                      help='Kiwi connection user name')
    parser.add_option('-f', '--freq',
                      dest='frequency',
                      type='float', default=1000,
                      help='Frequency to tune to, in kHz')
    parser.add_option('-z', '--zoom',
                      dest='zoom', type='int', default=0,
                      help='Zoom level 0-14')
    parser.add_option('--station',
                      dest='station',
                      type='string', default=None,
                      help='Station ID to be appended to filename',)
    parser.add_option('--log', '--log-level', '--log_level', type='choice',
                      dest='log_level', default='warn',
                      choices=['debug', 'info', 'warn', 'error', 'critical'],
                      help='Log level: debug|info|warn(default)|error|critical')
    (opt, unused_args) = parser.parse_args()

    ## clean up OptionParser which has cyclic references
    parser.destroy()

    opt.rigctl_enabled = False
    opt.is_kiwi_tdoa = False
    opt.tlimit = None
    opt.no_api = True
    opt.nolocal = False
    opt.S_meter = -1
    opt.ADC_OV = None
    opt.freq_pbc = None
    opt.wf_cal = None
    opt.netcat = False
    opt.wideband = False

    FORMAT = '%(asctime)-15s pid %(process)5d %(message)s'
    logging.basicConfig(level=logging.getLevelName(opt.log_level.upper()), format=FORMAT)
    #if opt.log_level.upper() == 'DEBUG':
    #    gc.set_debug(gc.DEBUG_SAVEALL | gc.DEBUG_LEAK | gc.DEBUG_UNCOLLECTABLE)

    run_event = threading.Event()
    run_event.set()

    snd_queue,wf_queue = [Queue(),Queue()]
    snd_recorder = KiwiWorker(args=(KiwiSoundRecorder    (opt, snd_queue), opt, False, run_event))
    wf_recorder  = KiwiWorker(args=(KiwiWaterfallRecorder(opt, wf_queue),  opt, False, run_event))
    consumer     = Consumer(args=(opt,snd_queue,wf_queue,run_event))

    threads = [snd_recorder, wf_recorder, consumer]

    try:
        opt.start_time = time.time()
        opt.ws_timestamp = int(time.time() + os.getpid()) & 0xffffffff
        opt.idx = 0
        snd_recorder.start()

        opt.start_time = time.time()
        opt.ws_timestamp = int(time.time() + os.getpid()+1) & 0xffffffff
        opt.idx = 0
        wf_recorder.start()

        consumer.start()

        while run_event.is_set():
            time.sleep(.5)

    except KeyboardInterrupt:
        run_event.clear()
        join_threads(threads)
        print("KeyboardInterrupt: threads successfully closed")
    except Exception:
        print_exc()
        run_event.clear()
        join_threads(threads)
        print("Exception: threads successfully closed")

    #logging.debug('gc %s' % gc.garbage)

if __name__ == '__main__':
    #import faulthandler
    #faulthandler.enable()
    main()
# EOF
