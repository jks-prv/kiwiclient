import array
import codecs
import logging
import os
import struct
import sys
import time
import math
import cmath
import traceback
from optparse import OptionParser

from kiwi import KiwiSDRStream, KiwiTooBusyError, KiwiBadPasswordError

import png


# Known bugs and missing features:
# * No automatic LPM detection; useful when a station switches between 60 and 120
# * No automatic white level correction
# * When IQ input is used, the moving average would be beneficial to lower loise level


def dump_to_csv(filename, data, mode='a'):
    with open(filename, mode) as fp:
        for x in data:
            fp.write("%.6f," % x)
        fp.write("\n")


def clamp(x, xmin, xmax):
    if x < xmin:
        x = xmin
    if x > xmax:
        x = xmax
    return x

def norm_clamp(x, xmin, xmax):
    return (clamp(x, xmin, xmax) - xmin) / (xmax - xmin)

def fm_detect(X, prev, shift):
    vals = array.array('f')
    for x in X:
        y = shift + cmath.phase(x * prev.conjugate()) / math.pi
        vals.append(y)
        prev = x
    return vals


def dft_complex(input):
    width = len(input)
    output = []
    w1d = complex(0, -2 * math.pi / width)
    w1 = 0
    for k in range(width):
        X = 0
        w2d = cmath.exp(w1)
        w2 = complex(1, 0)
        for n in range(width):
            X += input[n] * w2
            w2 *= w2d
        output.append(X)
        w1 += w1d
    return output

def idft_complex(input):
    width = len(input)
    width_inv = 1.0 / width
    output = []
    w1d = complex(0, 2 * math.pi / width)
    w1 = 0
    for n in range(width):
        X = 0
        w2d = cmath.exp(w1)
        w2 = complex(1, 0)
        for k in range(width):
            X += input[k] * w2
            w2 *= w2d
        output.append(X * width_inv)
        w1 += w1d
    return output

def bitreverse_sort(input):
    output = list(input)

    half_length = len(input) // 2
    j = half_length
    for i in range(1, len(input) - 1):
        if i < j:
            t = output[j]
            output[j] = output[i]
            output[i] = t
        k = half_length
        while k <= j:
            j -= k
            k = k >> 1
        j += k

    return output

def log2(x):
    return math.frexp(x)[1] - 1

def fft_core(x):
    length = len(x)

    for l in range(1, log2(length) + 1):
        le = 1 << l
        le2 = le >> 1
        w = 2 * math.pi / le
        s = cmath.exp(complex(0, -w))
        u = complex(1, 0)
        for j in range(1, le2 + 1):
            for i in range(j - 1, length, le):
                o = i + le2
                t = x[o] * u
                x[o] = x[i] - t
                x[i] = x[i] + t
            u *= s

def fft_complex(input):
    x = bitreverse_sort(input)
    fft_core(x)
    return x

def ifft_complex(input):
    "Computes an inverse FFT transform for complex-valued input"
    x = bitreverse_sort(input)
    x = [ v.conjugate() for v in x ]
    fft_core(x)
    n_inv = 1.0 / len(x)
    x = [ v.conjugate() * n_inv for v in x ]
    return x

def power_db(input):
    nf = 1.0 / len(input)
    return [ 10 * math.log10(abs(x) * nf) for x in input ]

def peak_detect(data, thresh):
    data = array.array('f', data)
    peak_radius = 50
    peaks = []
    while True:
        peak_index = 0
        peak_value = data[peak_index]
        for i in range(1, len(data)):
            if peak_value < data[i]:
                peak_value = data[i]
                peak_index = i
        if peak_value < thresh:
            break
        peaks.append((peak_index, peak_value))
        for i in range(max(peak_index - peak_radius, 0), min(peak_index + peak_radius + 1, len(data))):
            data[i] = -999
    return peaks


class FMDetectorAtan2:
    def __init__(self):
        self._prev = complex(0)
    def process(self, samples):
        Y = array.array('f')
        prev = self._prev
        for x in samples:
            y = cmath.phase(x * prev.conjugate()) / math.pi
            Y.append(y)
            prev = x
        self._prev = prev
        return Y

class IQConverterDDC:
    """Convert audio samples to IQ: digital down-convert method"""
    def __init__(self, fc):
        "fc is the LO frequency divided by the sample rate"
        self._w = cmath.rect(1, -fc * 2 * math.pi)
        self._v = complex(1)
    def process(self, samples):
        Y = []
        for x in samples:
            Y.append(x * self._v)
            self._v *= self._w
        return Y

class IQConverterFFT:
    def __init__(self):
        pass
    def process(self, samples):
        X = fft_complex([ complex(x) for x in samples ])
        w = 1 + len(X) // 2
        Y = []
        for i in range(0, w):
            Y.append(X[i])
        for i in range(w, len(X)):
            Y.append(complex(1e-6))
        return ifft_complex(Y)


def interp_hermite(t, p0, p1, p2, p3):
    c0 = p1
    c1 = 0.5 * (p2 - p0)
    c2 = p0 - 2.5 * p1 + 2 * p2 - 0.5 * p3
    c3 = 0.5 * (p3 - p0) + 1.5 * (p1 - p2)
    return c0 + (t * (c1 + (t * (c2 + (t * c3)))))

class Interpolator:
    def __init__(self, factor):
        self._buffer = array.array('f')
        self._t = 0
        self.set_factor(factor)
    def set_factor(self, factor):
        self._dt = factor
    def extend(self, samples):
        for x in samples:
            self._buffer.append(float(x))
    def __iter__(self):
        return self
    def __next__(self):
        t_int = math.trunc(self._t)
        t_frac = self._t - t_int
        if t_int + 3 >= len(self._buffer):
            self._flush()
            raise StopIteration()
        self._t += self._dt
        return interp_hermite(t_frac, self._buffer[t_int], self._buffer[t_int + 1], self._buffer[t_int + 2], self._buffer[t_int + 3])
    def _flush(self):
        t_int = math.trunc(self._t)
        t_new = min(t_int, len(self._buffer))
        if t_new > 0:
            self._t -= t_new
            self._buffer = self._buffer[t_new:]

class FIRFilter:
    def __init__(self, kernel):
        self._kernel = kernel
        self._buffer = []
    def process(self, samples):
        self._buffer.extend(samples)
        Y = []
        i = 0
        while i + len(self._kernel) < len(self._buffer):
            y = 0
            for j in range(len(self._kernel)):
                y += self._buffer[i+j] * self._kernel[-j-1]
            Y.append(y)
            i += 1
        self._buffer = self._buffer[i:]
        return Y

def generate_sinc(fc, length):
    "Generates a sinc kernel"

    h = []
    w = 2 * math.pi * fc
    zf = (length - 1) / 2
    for i in range(0, length):
        x = i - zf
        if x == 0:
            h.append(w)
        else:
            h.append(math.sin(w * x) / x)
    return h

def generate_cosine_window_3(length, a, b, c, d):
    w = (2 * math.pi) / (length - 1)
    return [(a - b * math.cos(w * i) + c * math.cos(2 * w * i) -d * math.cos(3 * w * i)) for i in range(0, length)]

def generate_blackman_nuttall_window(length):
    return generate_cosine_window_3(length, 0.3635819, 0.4891775, 0.1365995, 0.0106411)

def apply_window(h, hw):
    if len(h) != len(hw):
        raise ValueError("vectors must have equal lengths")
    return [ h[i] * hw[i] for i in range(len(hw)) ]


def mapper_df_to_intensity(dfs, black_thresh, white_thresh):
    for x in dfs:
        yield norm_clamp(x, black_thresh, white_thresh)

class Histogram:
    def __init__(self, bins, xmin, xmax):
        self._min = xmin
        self._max = xmax
        self._bins = [ 0 for i in range(bins) ]
    def put(self, x):
        x = clamp(x, self._min, self._max)
        x = (x - self._min) / (self._max - self._min)
        i = int(x * (len(self._bins) - 1))
        self._bins[i] += 1
    def clear(self):
        for i in range(len(self._bins)):
            self._bins[i] = 0
    def get(self):
        s = 1.0 / sum(self._bins)
        return [ x * s for x in self._bins ]


# Let them have a name
RADIOFAX_WHITE_FREQ = 2300
RADIOFAX_BLACK_FREQ = 1500
RADIOFAX_STARTSTOP_FREQ = 1900
RADIOFAX_IOC576_START_TONE = 300
RADIOFAX_IOC288_START_TONE = 675
RADIOFAX_STOP_TONE = 450

class KiwiFax(KiwiSDRStream):
    def __init__(self, options):
        super(KiwiFax, self).__init__()
        self._options = options
        self._reader = True
        self._options.idx = 0
        self._options.stats = True
        self._type = 'SND'
        self._start_time = time.time()

        self._ioc = options.ioc
        self._lpm = options.lpm

        self._state = 'idle'

        self._use_iq = options.iq_stream

        self._iqconverter = None
        self._iqfir = None
        self._tuning_offset = options.force_offset
        self._ss_window_size = 4096
        self._startstop_buffer = []
        self._startstop_score = 0

        self._prevX = complex(0)
        self._phasing_count = 0
        self._resampler = None
        self._line_scale_factor = 1.0 - 1e-6 * options.sr_coeff
        self._rows = []
        self._pixel_buffer = array.array('f')
        # TODO: compute instead of hardcoding
        self._pixels_per_line = 1809
        # NOTE: Kyodo pages are ~8500px
        self._max_height = options.max_height

        self._histoa = Histogram(200, -0.1, +0.1)
        self._histob = Histogram(257, 0, 1)

        self._new_roll()
        if options.force:
            self._switch_state('printing')

    def _switch_state(self, new_state):
        logging.info("Switching to: %s", new_state)
        self._state = new_state
        if new_state == 'idle':
            self._startstop_score = 0
            self._noise_score = 0
            self._histoa.clear()
            self._histob.clear()
        elif new_state == 'starting':
            pass
        elif new_state == 'phasing':
            self._new_roll()
            self._phasing_count = 0
        elif new_state == 'printing':
            self._startstop_score = 0
        elif new_state == 'stopping':
            pass

    def _setup_rx_params(self):
        df = 1500
        if self._use_iq:
            # Tuned to the baseband
            bw = (RADIOFAX_WHITE_FREQ - RADIOFAX_BLACK_FREQ) / 2 + df
            self.set_mod('iq', -bw, +bw, self._options.frequency)
        else:
            # Tuned to USB (-1900 Hz)
            self.set_mod('usb', RADIOFAX_BLACK_FREQ - df, RADIOFAX_WHITE_FREQ + df, self._options.frequency - 1.9)
        # TODO: figure out proper AGC parameters
        self.set_agc(True)
        self.set_name('kiwifax.py')
        # self.set_geo('Antarctica')

    def _on_sample_rate_change(self):
        sample_rate = float(self._sample_rate)
        # Precompute everything that depends on the SR
        self._iqconverter = IQConverterDDC(RADIOFAX_STARTSTOP_FREQ / sample_rate)
        filter_width = 450 # Hz
        filter_taps = 17
        self._iqfir = FIRFilter(apply_window(generate_sinc(filter_width / sample_rate, filter_taps), generate_blackman_nuttall_window(filter_taps)))
        # Start/stop detection params
        resolution = sample_rate / self._ss_window_size
        self._bin_size = resolution
        self._startstop_center_bin = self._ss_window_size // 2 + 0
        self._start576_delta = int(RADIOFAX_IOC576_START_TONE / resolution)
        self._start288_delta = int(RADIOFAX_IOC288_START_TONE / resolution)
        self._stop_delta = int(RADIOFAX_STOP_TONE / resolution)
        self._ss_width = int(0.5 * (RADIOFAX_WHITE_FREQ - RADIOFAX_STARTSTOP_FREQ) / resolution)
        logging.info("Start/stop center bin: %d; width: %d", self._startstop_center_bin, self._ss_width)
        logging.info("Start side bins: %d/%d; stop side bins: %d/%d",
            self._startstop_center_bin+self._start576_delta, self._startstop_center_bin-self._start576_delta,
            self._startstop_center_bin+self._stop_delta, self._startstop_center_bin-self._stop_delta)
        # NOTE: tone width is halved -- it should be precise anyway
        self._ss_tone_width = int(0.5 * 0.5 * (RADIOFAX_STOP_TONE - RADIOFAX_IOC576_START_TONE) / resolution)
        # Pixel output params
        samples_per_line = sample_rate * 60.0 / self._lpm
        resample_factor = (samples_per_line / self._pixels_per_line) * self._line_scale_factor
        self._resampler = Interpolator(resample_factor)
        logging.info("Resampling factor: %f", resample_factor)
        contrast = 0.01
        brightness = 0.02
        shift = 0.00
        self._white_level = (2 * (RADIOFAX_WHITE_FREQ - RADIOFAX_STARTSTOP_FREQ) / sample_rate) - contrast - brightness + shift
        self._black_level = (2 * (RADIOFAX_BLACK_FREQ - RADIOFAX_STARTSTOP_FREQ) / sample_rate) + contrast + shift
        self._fc_factor = 2 * self._bin_size / sample_rate

    def _process_audio_samples(self, seq, samples, rssi):
        k = 1 / 32768.0
        samples = [ x * k for x in samples ]
        samples = self._iqconverter.process(samples)
        self._process_samples(seq, samples, rssi)

    def _process_iq_samples(self, seq, samples, rssi, gps):
        k = 1 / 32768.0
        samples = [ x * k for x in samples ]
        self._process_samples(seq, samples, rssi)

    def _process_samples(self, seq, samples, rssi):
        logging.info('Block: %08x, RSSI: %04d %s', seq, rssi, self._state)
        if not self._use_iq:
            samples = self._iqfir.process(samples)
        self._process_startstop(samples)
        self._process_pixels(samples)

    def _startstop_score_update(self, updown):
        if updown:
            if self._startstop_score < 10:
                self._startstop_score += 1
        elif self._startstop_score > 0:
            self._startstop_score -= 2
            if self._startstop_score < 0:
                self._startstop_score = 0

    def _process_startstop(self, samples):
        self._startstop_buffer.extend(samples)
        # Snip out a window for start/stop processing
        # Window size defines the overall size of the window
        # Window shift defines how many samples are discarded after each iteration
        # This allows for overlapping FFTs thus increasing temporal resolution
        window_shift = self._ss_window_size // 2
        while len(self._startstop_buffer) >= self._ss_window_size:
            window = self._startstop_buffer[:self._ss_window_size]
            self._startstop_buffer = self._startstop_buffer[window_shift:]
            self._process_startstop_piece(window)

    def _process_startstop_piece(self, samples):
        # Compute the power spectrum
        samples = fft_complex(samples)
        P = power_db(samples)
        # DC "removal" for IQ
        if self._use_iq:
            P[0] = P[1]
        # Panoramize
        P1 = P[len(P)//2:]
        P1.extend(P[:len(P)//2])
        P = P1
        # DUMP POINT
        if self._options.dump_spectra and self._state != 'idle':
            dump_to_csv(self._output_name + '-ss.csv', P)
        # Assume noise floor is the median value + 5dB
        Px = P[2048-425:2048+425]
        Psorted = sorted(Px)
        nf_level = Psorted[len(Psorted) // 2] + 5.0
        pk_level = Psorted[-1]
        peaks = peak_detect(P, nf_level + 10)
        logging.info("Peaks: [%s]", ' '.join([ '%04d:%+05.1f' % (x[0], x[1]) for x in peaks ]))
        # For each peak, test if it's the one around the start/stop middle freq
        # For 4096-wide FFT: W=981 B=640 S=810 Start576=[682,939], Stop=[618,1002]
        detect_startstop = False
        detect_start576L = False
        detect_start576H = False
        detect_stopL = False
        detect_stopH = False
        # Classify the peaks
        for peak_bin, peak_power in peaks:
            # Try to classify the peak
            # Don't apply tuning correction for the start/stop center peak
            if math.fabs(peak_bin - self._startstop_center_bin) < self._ss_width:
                # NOTE: If force started, this doesn't get triggered properly
                if self._state in ('idle', 'starting'):
                    self._tuning_offset = self._startstop_center_bin - peak_bin
                detect_startstop = True
            else:
                peak_bin_relative = peak_bin + self._tuning_offset - self._startstop_center_bin
                if math.fabs(peak_bin_relative - self._stop_delta) < self._ss_tone_width:
                    detect_stopL = True
                if math.fabs(peak_bin_relative + self._stop_delta) < self._ss_tone_width:
                    detect_stopH = True
                if math.fabs(peak_bin_relative - self._start576_delta) < self._ss_tone_width:
                    detect_start576L = True
                if math.fabs(peak_bin_relative + self._start576_delta) < self._ss_tone_width:
                    detect_start576H = True
        detect_start576 = detect_startstop and detect_start576L and detect_start576H
        detect_stop = detect_startstop and detect_stopL and detect_stopH
        if self._state in ('idle', 'starting'):
            self._startstop_score_update(detect_start576)
        else:
            self._startstop_score_update(detect_stop)

        logging.info("NF=%05.1f PK=%05.1f  TO=%+04d/%+06.2fHz SS=%02d %s%s%s%s%s",
            nf_level, pk_level, self._tuning_offset, self._tuning_offset * self._bin_size,
            self._startstop_score,
            "sS"[detect_startstop], '-5'[detect_start576L], '-5'[detect_start576H], "xX"[detect_stopL], "xX"[detect_stopH])
        # Decide
        if self._state == 'idle':
            if self._startstop_score >= 10:
                logging.critical("START DETECTED")
                self._switch_state('starting')
        elif self._state == 'starting':
            if self._startstop_score < 3:
                self._switch_state('phasing')
        elif self._state == 'printing':
            if self._startstop_score >= 10:
                logging.critical("STOP DETECTED")
                self._switch_state('stopping')
        elif self._state == 'stopping':
            if self._startstop_score < 3:
                self._flush_rows()
                self._switch_state('idle')

    def _new_roll(self):
        self._rows = []
        ts = time.strftime('%Y%m%dT%H%MZ', time.gmtime())
        self._output_name = '%s_%d' % (ts, int(self._options.frequency * 1000))
        if self._options.station:
            self._output_name += '_' + self._options.station

    def _process_pixels(self, samples):
        if not self._state in ('phasing', 'printing', 'stopping'):
            return
        shift = self._tuning_offset * self._fc_factor
        pixels = fm_detect(samples, self._prevX, shift)
        self._prevX = samples[-1]
        # DUMP POINT
        if self._options.dump_pixels:
            dump_to_csv(self._output_name + '-px.csv', pixels)
        for x in pixels:
            self._histoa.put(x)
        # Remap the detected region into [0,1)
        pixels = array.array('f', mapper_df_to_intensity(pixels, self._black_level, self._white_level))
        for x in pixels:
            self._histob.put(x)
        # Scale and adjust pixel rate
        self._resampler.extend(pixels)
        self._pixel_buffer.extend(self._resampler)

        if self._state == 'phasing':
            self._process_phasing()
        else:
            # Cut into rows of pixels
            while len(self._pixel_buffer) >= self._pixels_per_line:
                row = self._pixel_buffer[:self._pixels_per_line]
                self._pixel_buffer = self._pixel_buffer[self._pixels_per_line:]
                self._process_row(row)

    def _process_phasing(self):
        # Count attempts at phasing to avoid getting stuck
        self._phasing_count += 1
        # Skip 3-4 lines; it seems phasing is not reliable when started right away
        if self._phasing_count <= 3:
            self._pixel_buffer = self._pixel_buffer[self._pixels_per_line:]
            return
        if self._phasing_count >= 100:
            logging.error("Phasing failed! Starting anyway")
            self._switch_state('printing')
            return
        # Do a moving average of the pixel intensity
        phasing_pulse_size = 90
        i = 0
        while i + phasing_pulse_size < len(self._pixel_buffer):
            s = 0
            for j in range(i, i + phasing_pulse_size):
                s += clamp(self._pixel_buffer[j], 0, 1)
            s /= phasing_pulse_size
            if s >= 0.85:
                self._pixel_buffer = self._pixel_buffer[i + phasing_pulse_size * 3 // 4:]
                logging.info("Phasing OK")
                self._switch_state('printing')
                break
            i += 1
        else:
            self._pixel_buffer = self._pixel_buffer[max(0, i - phasing_pulse_size):]

    def _process_row(self, row):
        pixels = array.array('B')
        for x in row:
            pixels.append(int(clamp(x, 0, 1) * 255))
        self._rows.append(pixels)
        if len(self._rows) % 16:
            return
        self._flush_rows()
        if len(self._rows) >= self._max_height:
            logging.info("Length exceeded; cutting the paper")
            self._switch_state('idle')

    def _flush_rows(self):
        if not self._rows:
            return
        while True:
            with open(self._output_name + '.png', 'wb') as fp:
                try:
                    png.Writer(len(self._rows[0]), len(self._rows), greyscale=True).write(fp, self._rows)
                    break
                except KeyboardInterrupt:
                    pass
        # DUMP POINT
        if self._options.dump_histo:
            dump_to_csv(self._output_name + '-hh.csv', self._histoa.get(), 'w')
            dump_to_csv(self._output_name + '-hh.csv', self._histob.get(), 'a')


KNOWN_CORRECTION_FACTORS = {
    'kiwisdr.northlandradio.nz:8073': {
        11030.00: -11.0,
    },
    'travelx.org:8073': { # +7.0
        7795.00: +3.0,
        9165.00: -5.0,
        13988.50: +3.0,
        16971.00: +4.0,
    },
    'travelx.org:8074': {
        7795.00: +0.0,
        9165.00: -14.0,
    },
    'reute.dyndns-remote.com:8073': {
        7880.00: -15.0,
        13882.50: -15.0,
    },
    'sarloutca.ddns.net:8073': {
        7880.00: -11.0,
        13882.50: -11.0,
    },
    'szsdr.ddns.net:8073': {
        9165.00: -11.0,
    },
    '72.130.191.200:8073': {
        9982.50: -13.0,
        11090.00: -13.0,
        16135.00: -13.0,
    }
}

def main():
    #sys.stdout = codecs.getwriter('utf-8')(sys.stdout)

    parser = OptionParser()
    parser.add_option('-k', '--socket-timeout', '--socket_timeout',
                      dest='socket_timeout', type='int', default=10,
                      help='Timeout(sec) for sockets')
    parser.add_option('-s', '--server-host', '--server_host',
                      dest='server_host', type='string',
                      default='localhost', help='server host')
    parser.add_option('-p', '--server-port', '--server_port',
                      dest='server_port', type='int',
                      default=8073, help='server port (default 8073)')
    parser.add_option('--pw', '--password',
                      dest='password', type='string', default='',
                      help='Kiwi login password (if required)')
    parser.add_option('--tlimit-pw', '--tlimit-password',
                      dest='tlimit_password', type='string', default='',
                      help='Connect time limit exemption password')
    parser.add_option('-q', '--iq',
                      dest='iq_mode',
                      action='store_true', default=False,
                      help='IQ data mode')

    parser.add_option('-f', '--freq',
                      dest='frequency',
                      type='float', default=4610,
                      help='Frequency to tune to, in kHz (will be tuned down by 1.9kHz)')
    parser.add_option('--station', '--station',
                      dest='station',
                      type='string', default=None,
                      help='Station ID to be appended to file names')
    parser.add_option('-F', '--force-start',
                      dest='force',
                      action='store_true', default=False,
                      help='Force the decoding without waiting for start tone or phasing')
    parser.add_option('--force-offset', '--force_offset',
                      dest='force_offset',
                      type='int', default=0,
                      help='When force decoding, apply this tuning offset (bins).')
    parser.add_option('-i', '--ioc',
                      dest='ioc',
                      type='int', default=576,
                      help='Index of cooperation; default: 576.')
    parser.add_option('-l', '--lpm',
                      dest='lpm',
                      type='int', default=120,
                      help='Lines per minute; default: 120.')
    parser.add_option('--sr-coeff', '--sr_coeff',
                      dest='sr_coeff',
                      type='float', default=0,
                      help='Sample frequency correction, ppm; positive if the lines are too short; negative otherwise')
    parser.add_option('--max-height', '--max_height',
                      dest='max_height',
                      type='int', default=2300,
                      help='Maximum page height; default: 2300.')
    parser.add_option('--dump-spectra', '--dump-spectra',
                      dest='dump_spectra',
                      action='store_true', default=False,
                      help='Dump block spectra to a CSV file')
    parser.add_option('--dump-pixels', '--dump-pixels',
                      dest='dump_pixels',
                      action='store_true', default=False,
                      help='Dump row pixels to a CSV file')
    parser.add_option('--dump-histo', '--dump_histo',
                      dest='dump_histo',
                      action='store_true', default=False,
                      help='Dump pixel intensity histograms to a CSV file')
    parser.add_option('--iq-stream', '--iq_stream',
                      dest='iq_stream',
                      action='store_true', default=False,
                      help='EXPERIMENTAL: use IQ stream instead of audio')
    parser.add_option('--tlimit', '--time-limit',
                      dest='tlimit',
                      type='float', default=None,
                      help='Record time limit in seconds')
    parser.add_option('--OV',
                      dest='ADC_OV',
                      default=False,
                      action='store_true',
                      help='Print "ADC OV" message when Kiwi ADC is overloaded')

    (options, unused_args) = parser.parse_args()
    options.ws_timestamp = int(time.time() + os.getpid()) & 0xffffffff
    options.netcat = False
    options.nolocal = False
    options.S_meter = -1
    options.rigctl_enabled = False
    options.freq_pbc = None
    options.wf_cal = None
    options.wideband = False

    # Setup logging
    fmtr = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', '%Y%m%dT%H%MZ')
    fmtr.converter = time.gmtime
    fh = logging.FileHandler('log_%s_%d_%d.log' % (options.server_host, options.server_port, int(options.frequency * 1000)))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmtr)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmtr)
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)
    #rootLogger.setLevel(logging.DEBUG)
    rootLogger.addHandler(fh)
    rootLogger.addHandler(ch)

    logging.critical('* * * * * * * *')
    logging.critical('Logging started')

    if options.sr_coeff == 0:
        server_identity = '%s:%d' % (options.server_host, options.server_port)
        try:
            coeffs = KNOWN_CORRECTION_FACTORS[server_identity]
            known_coeff = coeffs[options.frequency]
            options.sr_coeff = known_coeff
            logging.info('Applying known correction %f for host %s', known_coeff, server_identity)
        except KeyError:
            pass

    while True:
        recorder = KiwiFax(options)

        # Connect
        try:
            recorder.connect(options.server_host, options.server_port)
            recorder.open()
        except KeyboardInterrupt:
            break
        except Exception as e:
            traceback.print_exc()
            print("Failed to connect, sleeping and reconnecting")
            time.sleep(15)
            continue
        # Record
        try:
            while True:
              recorder.run()
            break
        except (KiwiTooBusyError, KiwiBadPasswordError):
            print("Server too busy now, sleeping and reconnecting")
            time.sleep(15)
            continue
        except Exception as e:
            traceback.print_exc()
            break
    recorder.close()
    print("exiting")


if __name__ == '__main__':
    main()
# EOF
