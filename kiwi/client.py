#!/usr/bin/env python

import array, os
import logging
import socket
import struct
import time
import numpy as np

try:
    import urllib.parse as urllib
except ImportError:
    import urllib

import sys
if sys.version_info > (3,):
    buffer = memoryview
    def bytearray2str(b):
        return b.decode('ascii')
else:
    def bytearray2str(b):
        return str(b)

import json
import mod_pywebsocket.common
from mod_pywebsocket._stream_base import ConnectionTerminatedException
from mod_pywebsocket.stream import Stream, StreamOptions
from .wsclient import ClientHandshakeProcessor, ClientRequest

#
# IMAADPCM decoder
#

stepSizeTable = (
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34,
    37, 41, 45, 50, 55, 60, 66, 73, 80, 88, 97, 107, 118, 130, 143,
    157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449, 494,
    544, 598, 658, 724, 796, 876, 963, 1060, 1166, 1282, 1411, 1552,
    1707, 1878, 2066, 2272, 2499, 2749, 3024, 3327, 3660, 4026,
    4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493, 10442,
    11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623,
    27086, 29794, 32767)

indexAdjustTable = [
    -1, -1, -1, -1,  # +0 - +3, decrease the step size
     2, 4, 6, 8,     # +4 - +7, increase the step size
    -1, -1, -1, -1,  # -0 - -3, decrease the step size
     2, 4, 6, 8      # -4 - -7, increase the step size
]


def clamp(x, xmin, xmax):
    if x < xmin:
        return xmin
    if x > xmax:
        return xmax
    return x

class ImaAdpcmDecoder(object):
    def __init__(self):
        self.index = 0
        self.prev = 0

    def preset(self, index, prev):
        self.index = index
        self.prev = prev

    def _decode_sample(self, code):
        #logging.debug("%d|%d" % (self.index, len(stepSizeTable)-1))
        step = stepSizeTable[self.index]
        self.index = clamp(self.index + indexAdjustTable[code], 0, len(stepSizeTable) - 1)
        difference = step >> 3
        if ( code & 1 ):
            difference += step >> 2
        if ( code & 2 ):
            difference += step >> 1
        if ( code & 4 ):
            difference += step
        if ( code & 8 ):
            difference = -difference
        sample = clamp(self.prev + difference, -32768, 32767)
        self.prev = sample
        return sample

    def decode(self, data):
        fcn = ord if isinstance(data, str) else lambda x : x
        samples = array.array('h')
        for b in map(fcn, data):
            sample0 = self._decode_sample(b & 0x0F)
            sample1 = self._decode_sample(b >> 4)
            samples.append(sample0)
            samples.append(sample1)
        return samples

#
# KiwiSDR WebSocket client
#

class KiwiError(Exception):
    pass
class KiwiTooBusyError(KiwiError):
    pass
class KiwiRedirectError(KiwiError):
    pass
class KiwiDownError(KiwiError):
    pass
class KiwiCampError(KiwiError):
    pass
class KiwiBadPasswordError(KiwiError):
    pass
class KiwiNoMultipleConnectionsError(KiwiError):
    pass
class KiwiTimeLimitError(KiwiError):
    pass
class KiwiServerTerminatedConnection(KiwiError):
    pass
class KiwiUnknownModulation(KiwiError):
    pass

class KiwiSDRStreamBase(object):
    """KiwiSDR WebSocket stream base client."""

    def __init__(self):
        self._socket = None
        self._decoder = None
        self._sample_rate = None
        self._version_major = None
        self._version_minor = None
        self._kiwi_version = None
        self._modulation = None
        self._stereo = False
        self._num_channels = 1
        self._lowcut = 0
        self._highcut = 0
        self._freq = 0
        self._stream = None

    def get_mod(self):
        return self._modulation

    def get_lowcut(self):
        return self._lowcut

    def get_highcut(self):
        return self._highcut

    def get_frequency(self):
        return self._freq

    def connect(self, host, port):
        # self._prepare_stream(host, port, 'SND')
        pass

    def _process_message(self, tag, body):
        logging.warn('Unknown message tag: %s' % tag)
        logging.warn(repr(body))

    def _prepare_stream(self, host, port, which):
        self._stream_name = which
        self._socket = socket.create_connection(address=(host, port), timeout=self._options.socket_timeout)
        uri = '%s/%d/%s%s' % ('/wb' if self._options.wideband else '', self._options.ws_timestamp, which, '?camp' if self._camp_chan != -1 else '')
        logging.debug('uri=<%s>' % uri)
        handshake = ClientHandshakeProcessor(self._socket, host, port)
        handshake.handshake(uri)

        request = ClientRequest(self._socket)
        request.ws_version = mod_pywebsocket.common.VERSION_HYBI13

        stream_option = StreamOptions()
        stream_option.mask_send = True
        stream_option.unmask_receive = False

        self._stream = Stream(request, stream_option)

    def _send_message(self, msg):
        if msg != 'SET keepalive':
            # stop sending commands back to Kiwi after receiving "SET monitor"
            if self._camping:
                return
            logging.debug("send SET (%s) \"%s\"", self._stream_name, msg)
        self._stream.send_message(msg)

    def _set_auth(self, client_type, password='', tlimit_password=''):
        if tlimit_password != '':
            if password == '':
                ## when the password is empty set it to '#' in order to correctly parse tlimit_password
                ## note that '#' as a password is being ignored by the kiwi server
                password = '#'
            self._send_message('SET auth t=%s p=%s ipl=%s' % (client_type, password, tlimit_password))
        else:
            self._send_message('SET auth t=%s p=%s' % (client_type, password))

    def set_name(self, name):
        self._send_message('SET ident_user=%s' % (name))

    def set_geo(self, geo):
        self._send_message('SET geo=%s' % (geo))

    def _set_keepalive(self):
        self._send_message('SET keepalive')

    def _process_ws_message(self, message):
        tag = bytearray2str(message[0:3])
        body = message[3:]
        self._process_message(tag, body)


SND_FLAG_ADC_OVFL      = 0x02
SND_FLAG_STEREO        = 0x08
SND_FLAG_COMPRESSED    = 0x10
SND_FLAG_LITTLE_ENDIAN = 0x80

class KiwiSDRStream(KiwiSDRStreamBase):
    """KiwiSDR WebSocket stream client."""

    def __init__(self, *args, **kwargs):
        super(KiwiSDRStream, self).__init__()
        self._decoder = ImaAdpcmDecoder()
        self._sample_rate = None
        self._version_major = None
        self._version_minor = None
        self._kiwi_version = None
        self._modulation = None
        self._stereo = False
        self._num_channels = 1
        self._lowcut = 0
        self._highcut = 0
        self._freq = 0
        self._compression = True
        self._gps_pos = [0,0]
        self._s_meter_avgs = self._s_meter_cma = 0
        self._s_meter_valid = False
        self._tot_meas_count = self._meas_count = 0
        self._stop = False
        self._need_nl = False
        self._kiwi_foff = 0
        self._camp_chan = -1
        self._camping = False
        self._comp_set = False
        self._decoder_index = 0
        self._decoder_prev = 0

        self._default_passbands = {
            "am":  [ -4900, 4900 ],
            "amn": [ -2500, 2500 ],
            "amw": [ -6000, 6000 ],
            "sam": [ -4900, 4900 ],
            "sal": [ -4900,    0 ],
            "sau": [     0, 4900 ],
            "sas": [ -4900, 4900 ],
            "qam": [ -4900, 4900 ],
            "drm": [ -5000, 5000 ],
            "lsb": [ -2700, -300 ],
            "lsn": [ -2400, -300 ],
            "usb": [   300, 2700 ],
            "usn": [   300, 2400 ],
            "cw":  [   300,  700 ],
            "cwn": [   470,  530 ],
            "nbfm":[ -6000, 6000 ],
            "nnfm":[ -3000, 3000 ],
            "iq":  [ -5000, 5000 ]
        }

        self.MAX_FREQ = 30e3 ## in kHz
        self.MAX_ZOOM = 14
        self.WF_BINS  = 1024

    def connect(self, host, port):
        self._prepare_stream(host, port, self._type)

    def _remove_freq_offset(self, freq):
        foffset = 0
        if hasattr(self, '_freq_offset'):   # in case called from app where it isn't defined
            # logging.debug('#### freq_offset=%d kiwi_foff=%d' % (self._freq_offset, self._kiwi_foff))
            if (self._kiwi_foff != 0) and (self._freq_offset != 0) and (self._freq_offset != self._kiwi_foff):
                s = "The Kiwi's configured frequency offset of %.3f kHz conflicts with -o option frequency offset of %.3f kHz" % (self._kiwi_foff, self._freq_offset)
                raise Exception(s)
            foffset = self._freq_offset
        if self._kiwi_foff != 0:
            foffset = self._kiwi_foff

        fmin = foffset
        fmax = foffset + self.MAX_FREQ
        if freq < fmin or freq > fmax:
            s = "Current frequency offset not compatible with -f option frequency.\n-f option frequency must be between %.3f and %.3f kHz" % (fmin, fmax)
            raise Exception(s)
        return freq - foffset       # API requires baseband freq to always be used

    def set_mod(self, mod, lc, hc, freq):
        mod = mod.lower()
        self._modulation = mod
        self._stereo = mod in [ "iq", "drm", "sas", "qam" ]
        self._num_channels = 2 if self._stereo else 1
        logging.debug('set_mod: stereo=%d num_channels=%d' % (self._stereo, self._num_channels))
        baseband_freq = self._remove_freq_offset(freq)
        
        if lc == None or hc == None:
            if mod in self._default_passbands:
                lc = self._default_passbands[mod][0] if lc == None else lc
                hc = self._default_passbands[mod][1] if hc == None else hc
            else:
                raise KiwiUnknownModulation('"%s"' % mod)

        if self._options.freq_pbc and mod in [ "lsb", "lsn", "usb", "usn", "cw", "cwn" ]:
            pbc = (lc + (hc - lc)/2)/1000
            freq = freq - pbc
            baseband_orig = baseband_freq
            baseband_freq = baseband_freq - pbc
            logging.debug('set_mod: freq=%.2f pbc_offset=%.2f pbc_freq=%.2f' % (baseband_orig, pbc, baseband_freq - pbc))
        self._send_message('SET mod=%s low_cut=%d high_cut=%d freq=%.3f' % (mod, lc, hc, baseband_freq))
        self._lowcut = lc
        self._highcut = hc
        self._freq = freq

    def set_freq(self, freq):
        self._freq = freq
        mod    = self._options.modulation
        lp_cut = self._options.lp_cut
        hp_cut = self._options.hp_cut
        if mod == 'am' or mod == 'amn' or mod == 'amw':
            # For AM, ignore the low pass filter cutoff
            lp_cut = -hp_cut if hp_cut is not None else hp_cut
        self.set_mod(mod, lp_cut, hp_cut, self._freq)

    def set_agc(self, on=False, hang=False, thresh=-100, slope=6, decay=1000, gain=50):
        logging.debug('set_agc: on=%s hang=%s thresh=%d slope=%d decay=%d gain=%d' % (on, hang, thresh, slope, decay, gain))
        self._send_message('SET agc=%d hang=%d thresh=%d slope=%d decay=%d manGain=%d' % (on, hang, thresh, slope, decay, gain))

    def set_squelch(self, sq, thresh):
        self._send_message('SET squelch=%d max=%d' % (sq, thresh))

    def set_noise_blanker(self, gate, thresh):
        # nb_algo(1) = NB_STD, type(0) = NB_BLANKER, type(2) = NB_CLICK
        self._send_message('SET nb algo=1')     # NB: setting algo clears all enables
        self._send_message('SET nb type=0 param=0 pval=%d' % gate)
        self._send_message('SET nb type=0 param=1 pval=%d' % thresh)
        self._send_message('SET nb type=0 en=%d' % (1 if self._options.nb else 0))
        self._send_message('SET nb type=2 param=0 pval=1')
        self._send_message('SET nb type=2 param=1 pval=1')
        self._send_message('SET nb type=2 en=%d' % (1 if self._options.nb_test else 0))

    def set_de_emp(self, de_emp):
        self._send_message('SET de_emp=%d' % de_emp)

    def _set_ar_ok(self, ar_in, ar_out):
        self._send_message('SET AR OK in=%d out=%d' % (ar_in, ar_out))

    def _set_gen(self, freq, attn):
        self._send_message('SET genattn=%d' % (attn))
        self._send_message('SET gen=%d mix=%d' % (freq, -1))

    def _set_zoom_cf(self, zoom, cf_kHz):
        if self._kiwi_version >= 1.329:
            self._send_message('SET zoom=%d cf=%f' % (zoom, cf_kHz))
        else:
            (counter,start_frequency) = self.start_frequency_to_counter(cf_kHz - self.zoom_to_span(zoom)/2)
            self._send_message('SET zoom=%d start=%f' % (zoom, counter))

    def zoom_to_span(self, zoom):
        """return frequency span in kHz for a given zoom level"""
        assert(zoom >=0 and zoom <= self.MAX_ZOOM)
        return self.MAX_FREQ/2**zoom

    def start_frequency_to_counter(self, start_frequency):
        """convert a given start frequency in kHz to the counter value used in older 'SET start=' API needed before v1.329"""
        assert(start_frequency >= 0 and start_frequency <= self.MAX_FREQ)
        counter = round(start_frequency/self.MAX_FREQ * 2**self.MAX_ZOOM * self.WF_BINS)
        ## actual start frequency
        start_frequency = counter * self.MAX_FREQ / self.WF_BINS / 2**self.MAX_ZOOM
        return counter,start_frequency

    def _set_maxdb_mindb(self, maxdb, mindb):
        self._send_message('SET maxdb=%d mindb=%d' % (maxdb, mindb))

    def _set_snd_comp(self, comp):
        # ignore command line compression setting in camp mode because compression
        # is determined by audio stream flag
        if self._camp_chan == -1 and not self._camping:
            self._compression = comp
            self._send_message('SET compression=%d' % (1 if comp else 0))

    def _set_stats(self):
        self._send_message('SET STATS_UPD ch=0')

    def _set_wf_comp(self, comp):
        self._compression = comp
        self._send_message('SET wf_comp=%d' % (1 if comp else 0))

    def _set_wf_speed(self, speed):
        if speed == 0:
            speed = 1
        assert(speed >= 1 and speed <= 4)
        self._send_message('SET wf_speed=%d' % speed)

    def _set_wf_interp(self, interp):
        if interp == -1:
            interp = 13     # drop sampling + CIC compensation (Kiwi UI default)
        assert((interp >= 0 and interp <= 4) or (interp >=10 and interp <= 14))
        self._send_message('SET interp=%d' % interp)
    
    def _set_kiwi_version(self):
        if self._version_major is None or self._version_minor is None:
            return
        self._kiwi_version = float(self._version_major) + float(self._version_minor) / 1000.
        logging.info("Kiwi server version: %d.%d" % (self._version_major, self._version_minor))

    def _process_msg_param(self, name, value):
        prefix = "recv MSG (%s)" % (self._stream_name)

        if name == 'extint_list_json':
            value = urllib.unquote(value)

        if name == 'load_cfg':
            logging.debug("%s load_cfg: (cfg info not printed)" % prefix)
            d = json.loads(urllib.unquote(value))
            self._gps_pos = [float(x) for x in urllib.unquote(d['rx_gps'])[1:-1].split(",")[0:2]]
            if self._options.idx == 0:
                logging.info("GNSS position: lat,lon=[%+6.2f, %+7.2f]" % (self._gps_pos[0], self._gps_pos[1]))
            self._on_gnss_position(self._gps_pos)
            return
        elif name == 'load_dxcfg':
            logging.debug("%s load_dxcfg: (cfg info not printed)" % prefix)
            return
        elif name == 'load_dxcomm_cfg':
            logging.debug("%s load_dxcomm_cfg: (cfg info not printed)" % prefix)
            return
        elif name == 'camp':
            v = value.split(",")
            logging.debug("%s camp: okay=%s rx=%s" % (prefix, v[0], v[1]))
            return
        elif name == 'audio_camp':
            v = value.split(",")
            logging.debug("%s audio_camp: disconnect=%s isLocal=%s" % (prefix, v[0], v[1]))
            return
        elif name == 'antsw_AntennaDenySwitching':
            return
        #elif name.startswith('antsw_'):
        #    return
        else:
            if value is None:
                logging.debug("%s %s" % (prefix, name))
            else:
                logging.debug("%s %s: %s" % (prefix, name, value))

        # Handle error conditions
        if name == 'too_busy':
            raise KiwiTooBusyError('%s: all %s client slots taken' % (self._options.server_host, value))
        if name == 'redirect':
            raise KiwiRedirectError(urllib.unquote(value))
        if name == 'badp' and value == '1':
            raise KiwiBadPasswordError('%s: bad password' % self._options.server_host)
        if name == 'badp' and value == '5':
            raise KiwiNoMultipleConnectionsError('%s: no multiple connections from the same IP address' % self._options.server_host)
        if name == 'down':
            raise KiwiDownError('%s: server is down atm' % self._options.server_host)
        if name == 'camp_disconnect':
            raise KiwiCampError("%s: camped connection closed or doesn't exist" % self._options.server_host)

        # Handle data items
        if name == 'audio_rate':
            self._set_ar_ok(int(value), 44100)
        elif name == 'sample_rate':
            self._sample_rate = float(value)
            self._on_sample_rate_change()
            # Optional, but is it?..
            self.set_squelch(0, 0)
            self._set_gen(0, 0)
            # Required to get rolling
            self._setup_rx_params()
            # Also send a keepalive
            self._set_keepalive()
        elif name == 'monitor':
            if self._camp_chan != -1:
                self._send_message('SET MON_CAMP=%d' % self._camp_chan)
                self._compression = False
                self._camping = True
        elif name == 'bandwidth':
            self.MAX_FREQ = float(value)/1000       # allows e.g. 32 MHz Kiwis
        elif name == 'wf_setup':
            # Required to get rolling
            self._setup_rx_params()
            # Also send a keepalive
            self._set_keepalive()
        elif name == 'wf_cal':
            if self._options.wf_cal is None:
                self._options.wf_cal = int(value)
        elif name == 'version_maj':
            self._version_major = int(value)
            self._set_kiwi_version()
        elif name == 'version_min':
            self._version_minor = int(value)
            self._set_kiwi_version()
        elif name == 'ext_client_init':
            logging.info("ext_client_init(is_locked)=%s" % value)
            if value == "1":
                raise Exception("Only one DRM instance can be run at a time on this Kiwi")
            self._send_message('SET ext_no_keepalive')      # let server know not to expect async keepalive from us
            self._setup_rx_params()
        elif name == 'freq_offset':
            self._kiwi_foff = float(value)
        elif name == 'audio_adpcm_state':
            decoder_preset = value.split(",")
            self._decoder_index = int(decoder_preset[0])
            self._decoder_prev  = int(decoder_preset[1])

    def _process_message(self, tag, body):
        if tag == 'MSG':
            self._process_msg(bytearray2str(body[1:])) ## skip 1st byte
        elif tag == 'SND':
            #try:
            self._process_aud(body)
            #except Exception as e:
            #    logging.error(e)
            # Ensure we don't get kicked due to timeouts
            self._set_keepalive()
        elif tag == 'W/F':
            self._process_wf(body[1:]) ## skip 1st byte
            # Ensure we don't get kicked due to timeouts
            self._set_keepalive()
        elif tag == 'EXT':
            body = bytearray2str(body[1:])
            for pair in body.split(' '):
                if '=' in pair:
                    name, value = pair.split('=', 1)
                    self._process_ext(name, urllib.unquote(value))
                else:
                    name = pair
                    self._process_ext(name, None)
        else:
            logging.warn("unknown tag %s" % tag)
            pass

    def _process_msg(self, body):
        for pair in body.split(' '):
            if '=' in pair:
                name, value = pair.split('=', 1)
                self._process_msg_param(name, value)
            else:
                name = pair
                self._process_msg_param(name, None)

    def _process_aud(self, body):
        flags,seq, = struct.unpack('<BI', buffer(body[0:5]))
        smeter,    = struct.unpack('>H',  buffer(body[5:7]))
        data       = body[7:]
        rssi       = 0.1*smeter - 127
        ##logging.info("SND flags %2d seq %6d RSSI %6.1f len %d" % (flags, seq, rssi, len(data)))
        if self._options.ADC_OV and (flags & SND_FLAG_ADC_OVFL):
            print(" ADC OV")

        if self._camp_chan != -1:
            if flags & SND_FLAG_COMPRESSED:
                self._compression = True
                if self._comp_set == False:
                    logging.debug("CAMP decoder PRESET %d,%d" % (self._decoder_index, self._decoder_prev))
                    self._decoder.preset(self._decoder_index, self._decoder_prev)
                    self._comp_set = True

            else:
                self._decoder_index = self._decoder_prev = 0
                self._compression = False
                self._comp_set = False

        # first rssi is no good because first audio buffer is leftover from last time this channel was used
        if self._options.S_meter >= 0 and not self._s_meter_valid:
            # tlimit in effect if streaming RSSI
            self._start_time = time.time()
            self._start_sm_ts = time.gmtime()
            self._s_meter_valid = True
            if not self._options.sound:
                return
        else:

            # streaming
            if self._options.S_meter == 0 and self._options.sdt == 0:
                self._meas_count += 1
                self._tot_meas_count += 1
                ts = time.strftime('%d-%b-%Y %H:%M:%S UTC ', time.gmtime()) if self._options.tstamp else ''
                print("%sRSSI: %6.1f %d" % (ts, rssi, self._options.tstamp))
                if not self._options.sound:
                    return
            else:

                # averaging with optional dt
                if self._options.S_meter >= 0:
                    self._s_meter_cma = (self._s_meter_cma * self._s_meter_avgs) + rssi
                    self._s_meter_avgs += 1
                    self._s_meter_cma /= self._s_meter_avgs
                    self._meas_count += 1
                    self._tot_meas_count += 1
                    now = time.gmtime()
                    sec_of_day = lambda x: 3600*x.tm_hour + 60*x.tm_min + x.tm_sec
                    if self._options.sdt != 0:
                        interval = (self._start_sm_ts is not None) and (sec_of_day(now)//self._options.sdt != sec_of_day(self._start_sm_ts)//self._options.sdt)
                        meas_sec = float(self._meas_count)/self._options.sdt
                    else:
                        interval = False
                    if self._s_meter_avgs == self._options.S_meter or interval:
                        ts = time.strftime('%d-%b-%Y %H:%M:%S UTC ', now) if self._options.tstamp else ''
                        if self._options.stats and self._options.sdt:
                            print("%sRSSI: %6.1f %.1f meas/sec" % (ts, self._s_meter_cma, meas_sec))
                        else:
                            print("%sRSSI: %6.1f" % (ts, self._s_meter_cma))
                        if interval:
                            self._start_sm_ts = time.gmtime()
                        if self._options.sdt == 0:
                            self._stop = True
                        else:
                            self._s_meter_avgs = self._s_meter_cma = 0
                            self._meas_count = 0
                    if not self._options.sound:
                        return

        # in camp mode it's the stream we're camping on that can have arbitrary endedness 
        dtype = '<h' if (self._camping and flags & SND_FLAG_LITTLE_ENDIAN) else '>h'

        if self._camping:
            if flags & SND_FLAG_STEREO:     # stereo mode is never compressed
                gps = dict(zip(['last_gps_solution', 'dummy', 'gpssec', 'gpsnsec'], struct.unpack('<BBII', buffer(data[0:10]))))
                data = data[10:]
                fmt = "(CAMP stereo)"
                count = len(data) // 2
                samples = np.ndarray(count, dtype=dtype, buffer=data).astype(np.float32)
                cs      = np.ndarray(count//2, dtype=np.complex64)
                cs.real = samples[0:count:2]
                cs.imag = samples[1:count:2]
                self._process_iq_samples(seq, cs, rssi, gps, fmt)
            else:
                if self._compression:
                    comp = "comp"
                    sarray = self._decoder.decode(data)
                    count = len(sarray)
                    data = np.ndarray(count, dtype='int16', buffer=sarray)
                    count = len(data)
                else:
                    comp = "no-comp"
                    count = len(data) // 2

                samples = np.ndarray(count, dtype=dtype, buffer=data).astype(np.float32)
                cs      = np.ndarray(count, dtype=np.complex64)
                cs.real = samples[0:count:1]
                cs.imag = cs.real       # replicate mono audio in both channels
                fmt = ("(CAMP mono %s)" % comp) if self._options.netcat is True else None
                self._process_iq_samples(seq, cs, rssi, None, fmt)
        else:
            if self._stereo:     # stereo mode is never compressed
                gps = dict(zip(['last_gps_solution', 'dummy', 'gpssec', 'gpsnsec'], struct.unpack('<BBII', buffer(data[0:10]))))
                data = data[10:]
                if self._options.netcat is True and self._options.resample == 0:
                    count = len(data) // 2
                    samples = np.ndarray(count, dtype=dtype, buffer=data).astype(np.int16)
                else:
                    count = len(data) // 2
                    samples = np.ndarray(count, dtype=dtype, buffer=data).astype(np.float32)
                    cs      = np.ndarray(count//2, dtype=np.complex64)
                    cs.real = samples[0:count:2]
                    cs.imag = samples[1:count:2]
                    samples = cs
                fmt = "(NC stereo)" if self._options.netcat is True else None
                self._process_iq_samples(seq, samples, rssi, gps, fmt)
            else:
                if self._compression:
                    comp = "comp"
                    sarray = self._decoder.decode(data)
                    count = len(sarray)
                    samples = np.ndarray(count, dtype='int16', buffer=sarray)
                else:
                    comp = "no-comp"
                    count = len(data) // 2
                    samples = np.ndarray(count, dtype=dtype, buffer=data).astype(np.int16)
                fmt = ("(NC mono %s)" % comp) if self._options.netcat is True else None
                self._process_audio_samples(seq, samples, rssi, fmt)

    def _process_wf(self, body):
        x_bin_server,flags_x_zoom_server,seq, = struct.unpack('<III', buffer(body[0:12]))
        data = body[12:]
        #logging.info("W/F seq %d len %d" % (seq, len(data)))
        if self._options.netcat is True:
            return self._process_waterfall_samples_raw(seq, data)
        if self._compression:
            self._decoder.__init__()   # reset decoder each sample
            samples = self._decoder.decode(data)
            samples = samples[:len(samples)-10]   # remove decompression tail
        else:
            samples = np.ndarray(len(data), dtype='B', buffer=data)
        self._process_waterfall_samples(seq, samples)

    def _get_output_filename(self, *ext_arg):
        ext = ".wav" if len(ext_arg) == 0 else ext_arg[0]
        if ext == ".wav" and self._options.test_mode:
            return os.devnull
        station = '' if self._options.station is None else '_'+ self._options.station

        # if multiple connections specified but not distinguished via --station then use index
        if self._options.multiple_connections and self._options.station is None:
            station = '_%d' % self._options.idx
        if self._options.filename != '':
            filename = '%s%s%s' % (self._options.filename, station, ext)
        else:
            ts  = time.strftime('%Y%m%dT%H%M%SZ', self._start_ts)
            if self._camping:
                filename = '%s_camp_rx%d%s%s' % (ts, self._camp_chan, station, ext)
            else:
                filename = '%s_%d%s_%s%s' % (ts, int(self._freq * 1000), station, self._options.modulation, ext)
        if self._options.dir is not None:
            filename = '%s/%s' % (self._options.dir, filename)
        return filename

    def _on_gnss_position(self, position):
        pass

    def _on_sample_rate_change(self):
        pass

    def _process_audio_samples(self, seq, samples, rssi, fmt):
        pass

    def _process_iq_samples(self, seq, samples, rssi, gps, fmt):
        pass

    def _process_waterfall_samples(self, seq, samples):
        pass

    def _process_waterfall_samples_raw(self, seq, data):
        pass

    def _process_ext(self, name, value):
        pass
    
    def _close_func(self):
        pass

    def _setup_rx_params(self):
        if self._type == 'W/F':
            self._set_zoom_cf(0, 0)
            self._set_maxdb_mindb(-10, -110)
            self._set_wf_speed(1)       # 1 Hz
            self._set_wf_interp(13)     # drop sampling + CIC compensation (Kiwi UI default)
        if self._type == 'SND':
            self._set_mod('am', 100, 2800, 4625.0)
            self._set_agc(True)

    def _setup_no_api(self):
        if self._options.user != 'none':
            user = self._options.user
            if user == 'blank':
                user = ""
            if user == 'spaces':
                user = "   "
            if user == 'spaces2':
                user = "a b c"
            if user == 'bad':
                user = chr(1)
            if user == 'bad2':
                user = 'chr('+ chr(1) +'1)'
            self.set_name(user)
        if self._options.bad_cmd:
            self._send_message('SET xxx=0')
            self._send_message('SET yyy=0')

    def _writer_message(self):
        pass

    def open(self):
        if self._type == 'SND' or self._type == 'W/F' or self._type == 'EXT':
            # "SET options=" must be sent before auth
            if self._options.nolocal:
                self._send_message('SET options=1')
            self._set_auth('kiwi', self._options.password, self._options.tlimit_password)

    def close(self):
        if self._stream == None:
            return
        try:
            ## STATUS_GOING_AWAY does not make the stream to wait for a reply for the WS close request
            ## this is used because close_connection expects the close response from the server immediately
            self._stream.close_connection(mod_pywebsocket.common.STATUS_GOING_AWAY)
            self._socket.close()
        except Exception as e:
            logging.error('websocket close: "%s"' % e)

    def run(self):
        """Run the client."""
        if self._reader:
            try:
                received = self._stream.receive_message()
                if received is None:
                    self._socket.close()
                    raise KiwiServerTerminatedConnection('server closed the connection cleanly')
            except ConnectionTerminatedException:
                    logging.debug('ConnectionTerminatedException')
                    raise KiwiServerTerminatedConnection('server closed the connection unexpectedly')

            self._process_ws_message(received)
        else:
            msg = self._writer_message()
            self._stream.send_message(msg)

        tlimit = self._options.tlimit
        time_limit = tlimit != None and self._start_time != None and time.time() - self._start_time > tlimit
        if time_limit or self._stop:
            if self._need_nl:
                print("")
                self._need_nl = False
            if self._options.stats and self._tot_meas_count > 0 and self._start_time != None:
                print("%.1f meas/sec" % (float(self._tot_meas_count) / (time.time() - self._start_time)))
            raise KiwiTimeLimitError('time limit reached')

# EOF
