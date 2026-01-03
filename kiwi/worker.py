#!/usr/bin/env python3
## -*- python -*-

import logging, time
import threading
from traceback import print_exc

from .client import KiwiTooBusyError, KiwiRedirectError, KiwiTimeLimitError, KiwiServerTerminatedConnection
from .rigctld import Rigctld

from queue import Queue
q_stream = Queue()

class KiwiWorker(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None):
        super(KiwiWorker, self).__init__(group=group, target=target, name=name)
        self._recorder, self._options, self._reader, self._delay_run, self._run_event, self._camp_wait_event = args
        self._recorder._reader = self._reader
        self._recorder._camp_wait_event = self._camp_wait_event
        self._event = threading.Event()
        self._rigctld = None
        if self._options.rigctl_enabled:
            self._rigctld = Rigctld(self._recorder, self._options.rigctl_port, self._options.rigctl_address)

    def _do_run(self):
        return self._run_event.is_set()

    def run(self):
        self.connect_count = self._options.connect_retries
        self.busy_count = self._options.busy_retries
        if self._delay_run:     # let snd/wf get established first
            time.sleep(3)
        
        if not self._reader:
            # writer thread
            self._recorder._stream = q_stream.get()     # stream passed from reader thread
            while self._do_run():
                try:
                    self._recorder.run()
                except Exception as e:
                    logging.debug('writer Exception=%s' %e)
                    break
            #logging.debug('writer STOP')
            self._run_event.clear()     # tell all other threads to stop
            return

        # reader thread
        #logging.debug('reader PID=%d' % threading.get_native_id())
        while self._do_run():
            try:
                self._recorder.connect(self._options.server_host, self._options.server_port)
                q_stream.put(self._recorder._stream)    # pass stream to writer thread

            except Exception as e:
                logging.warn("Failed to connect, sleeping and reconnecting error='%s'" %e)
                if self._options.is_kiwi_tdoa:
                    self._options.status = 1
                    break
                self.connect_count -= 1
                if self._options.connect_retries > 0 and self.connect_count == 0:
                    break
                if self._options.connect_timeout > 0:
                    self._event.wait(timeout = self._options.connect_timeout)
                continue

            try:
                self._recorder.open()
                while self._do_run():
                    self._recorder.run()
                    # do things like freq changes while not receiving sound
                    if self._rigctld:
                        self._rigctld.run()
                #logging.debug('reader STOP')

            except KiwiServerTerminatedConnection as e:
                if self._options.no_api:
                    msg = ''
                else:
                    msg = ' Reconnecting after 5 seconds'
                logging.info("%s:%s %s.%s" % (self._options.server_host, self._options.server_port, e, msg))
                self._recorder.close()
                if self._options.no_api:    ## don't retry
                    break
                self._recorder._start_ts = None ## this makes the recorder open a new file on restart
                self._event.wait(timeout=5)
                continue

            except KiwiTooBusyError:
                if self._options.is_kiwi_tdoa:
                    self._options.status = 2
                    break
                self.busy_count -= 1
                if self._options.busy_retries > 0 and self.busy_count == 0:
                    break
                logging.warn("%s:%d Too busy now. Reconnecting after %d seconds"
                      % (self._options.server_host, self._options.server_port, self._options.busy_timeout))
                if self._options.busy_timeout > 0:
                    self._event.wait(timeout = self._options.busy_timeout)
                continue

            except KiwiRedirectError as e:
                prev = self._options.server_host +':'+ str(self._options.server_port)
                # http://host:port
                #        ^^^^ ^^^^
                uri = str(e).split(':')
                self._options.server_host = uri[1][2:]
                self._options.server_port = uri[2]
                logging.warn("%s Too busy now. Redirecting to %s:%s" % (prev, self._options.server_host, self._options.server_port))
                if self._options.is_kiwi_tdoa:
                    self._options.status = 2
                    break
                self._event.wait(timeout=2)
                continue

            except KiwiTimeLimitError:
                break

            except Exception as e:
                if self._options.is_kiwi_tdoa:
                    self._options.status = 1
                print_exc()
                break

        self._run_event.clear()   # tell all other threads to stop
        #logging.debug('run_event CLEAR')
        self._recorder.close()
        self._recorder._close_func()
        if self._rigctld:
            self._rigctld.close()
