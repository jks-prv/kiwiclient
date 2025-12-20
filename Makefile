#
# Example uses of kiwirecorder.py and kiwifax.py
#

PY = python3
#PY = python3.13
#PY = python3.12

# set global environment variables KIWI_HOST and KIWI_PORT to the Kiwi you want to work with
ifeq ($(KIWI_HOST)x,x)
    HOST = kiwisdr.local
else
    HOST = $(KIWI_HOST)
endif

ifeq ($(KIWI_PORT)x,x)
    PORT = 8073
else
    PORT = $(KIWI_PORT)
endif

ifeq ($(KIWI_FREQ)x,x)
    FREQ = 10000
else
    FREQ = $(KIWI_FREQ)
endif

ifeq ($(KIWI_MODE_PB)x,x)
    MODE_PB = -m am -L -5000 -H 5000
else
    MODE_PB = $(KIWI_MODE_PB)
endif

KREC = $(PY) -u kiwirecorder.py

HP = -s $(HOST) -p $(PORT)
H2 = -s $(HOST),$(HOST) -p $(PORT)
H8 = -s $(HOST),$(HOST),$(HOST),$(HOST),$(HOST),$(HOST),$(HOST),$(HOST) -p $(PORT)

F = -f $(FREQ)
F_PB = $F $(MODE_PB)


# process control help
UNAME = $(shell uname)
ifeq ($(UNAME),Darwin)
# on OS X (Darwin) there is no "interactive mode" for killall command, so use 'kp' BEFORE 'kill' to check
kp:
	killall -d -KILL Python
kill:
#	killall -v -KILL Python
	killall -v Python
else
kp kill:
	killall -r -i -s KILL Python
endif

ps:
	ps ax | grep -i kiwirecorder


# record WSPR audio to file
#
# "-f" frequency is dial frequency, i.e. WSPR center frequency minus passband center (BFO)
# e.g. 40m: cf = 7040.1, so if pb center = 750 then dial = 7040.1 - 0.750 = 7039.35
# NB: most WSPR programs use a pb center of 1500 Hz, not 750 which we use because we think it's easier to listen to

wspr:
	$(KREC) $(HP) --filename=wspr_40m -f 7039.35 --user=WSPR_40m -m iq -L 600 -H 900 --tlimit=110 --log_level=debug

# multiple connections
wspr2:
	$(KREC) $(HP2) --filename=wspr_40m,wspr_30m -f 7039.35,10139.45 --user=WSPR_40m,WSPR_30m -m iq -L 600 -H 900 --tlimit=110


# DRM
# With the "--ext=DRM" argument kiwirecorder requests the Kiwi to start the DRM extension and send
# decoded DRM audio instead of normal analog audio.
#
# 15785 Funklust
# 3965 Telediffusion de France
# "--ext-test" uses test file built-in to DRM extension

HP_DRM = $(HP)

F_DRM = $(F_PB)
#F_DRM = -f 15785
#F_DRM = -f 3965
#F_DRM = -f 5555 --ext-test

DRM_COMMON = --mode=drm -L -5000 -H 5000 --user=DRM-test --log-level=info
#DRM_COMMON = --mode=drm -L -5000 -H 5000 --user=DRM-test --log-level=debug
DRM = $(HP_DRM) $(F_DRM) $(DRM_COMMON) --tlimit=15

drm-info:
	@echo HP_DRM = $(HP_DRM)
	@echo F_DRM = $(F_DRM)

# write decoded DRM audio to a file
drm-snd: drm-info
	$(KREC) $(DRM) --tlimit=30 --snd --ext=DRM

snd-wf: drm-info
	$(KREC) $(DRM) --tlimit=15 --snd --wf --z 9 --speed 4 --quiet --wf-png --wf-auto --log-level=debug

# show DRM stats (--stats) without writing audio file (--test-mode --quiet)
drm-stats: drm-info
	$(KREC) $(DRM) --tlimit=30 --snd --test-mode --quiet --ext=DRM --stats --log-level=warn

# show DRM stats (--stats) with writing audio file
drm-snd-stats: drm-info
	$(KREC) $(DRM) --tlimit=30 --snd --ext=DRM --stats --log-level=warn

# record Kiwi waterfall to a .png file, manual setup of WF min/max (no DRM decoding involved here)
drm-wf: drm-info
	$(KREC) $(DRM) --tlimit=15 --mode=am --wf --z 9 --speed 4 --quiet --wf-png --mindb -130 --maxdb -60 --wf-auto

# record Kiwi waterfall to a .png file, auto setup of WF min/max (no DRM decoding involved here)
drm-wf-auto: drm-info
	$(KREC) $(DRM) --tlimit=15 --mode=am --wf --z 9 --speed 4 --quiet --wf-png --wf-auto

# record Kiwi audio, waterfall and stats
drm-snd-wf: drm-info
	$(KREC) $(DRM) --tlimit=30 --snd --quiet --wf --z 9 --speed 4 --wf-png --wf-auto --ext=DRM --stats --log-level=warn


# DRM testing

drm-828:
	$(KREC) $(DRM) -s newdelhi.twrmon.net -f 828 --filename=Delhi.828.12k.iq
drm-1368:
	$(KREC) $(DRM) -s newdelhi.twrmon.net -f 1368 --filename=Delhi.1368.12k.iq
drm-621:
	$(KREC) $(DRM) -s bengaluru.twrmon.net -f 621 --filename=Bengaluru.621.12k.iq
drm-1044:
	$(KREC) $(DRM) -s 182.237.12.150.twrmon.net -f 1044 --filename=Bdq.1044.12k.iq
drm-9620:
#	$(KREC) $(DRM) -s emeraldsdr.ddns.net -f 9620 --filename=AIR.9620.12k.iq --mode=iq --tlimit=10 --log-level=info
#	$(KREC) $(DRM) -s emeraldsdr.ddns.net -f 9620 --filename=AIR.9620.12k.drm --ext=DRM --snd --s-meter=0 --sdt-sec=1 --tlimit=30 --log-level=info --ts
	$(KREC) -s emeraldsdr.ddns.net -p 8073 -f 9620 -L -5000 -H 5000 --mode=drm --ext=DRM --snd --user=DRM-test --filename=AIR.9620.12k.drm --s-meter=0 --sdt-sec=1 --tlimit=30 --timestamp --log-level=info
drm-5910:
	$(KREC) -s df0twn.dnsuser.de -p 8073 -f 5910 -L -5000 -H 5000 --mode=drm --ext=DRM --snd --user=DRM-test --filename=RRI.5910.12k.drm --s-meter=0 --sdt-sec=1 --tlimit=30 --timestamp --log-level=info

drm-bug:
#	$(KREC) $(DRM) --tlimit=40 --test-mode --snd --wf --z 5
#	$(KREC) $(DRM) --tlimit=4 --test-mode
#	$(KREC) $(DRM) --tlimit=10 --test-mode --wf --z 5
#	$(KREC) $(DRM) --tlimit=10 --test-mode --ext=DRM --wf --z 5
#	$(KREC) $(DRM) --tlimit=10 --test-mode --ext=DRM
	$(KREC) $(DRM) --tlimit=60 --test-mode --quiet --snd --ext=DRM
#	$(KREC) $(DRM) --tlimit=10 --test-mode --ext=DRM --nolocal


# see if Dream works using a real-mode stream (it does)
# requires a Kiwi in 3-channel mode (20.25 kHz) to accomodate a 10 kHz wide USB passband
dream_real:
	$(KREC) $(HP) -f $(FREQ) -m usb -L 0 -H 10000 --ncomp


# FAX
# has both real and IQ mode decoding

fax:
	$(PY) kiwifax.py $(HP) -f $(FREQ) -F
faxiq:
	$(PY) kiwifax.py $(HP) -f $(FREQ) -F --iq-stream


# Two separate IQ files recording in parallel
HOST_IQ1 = 
HOST_IQ2 = 

two:
	$(KREC) -s $(HOST_IQ1),$(HOST_IQ2) -p ($PORT) -f 77.5,60 --station=DCF77,MSF -m iq -L -5000 -H 5000


# real & iq mode file
# Should playback using standard .wav file player

re real:
	$(KREC) $(HP) $(F_PB) --tlimit=10 --log-level=debug
rere:
	$(KREC) $(HP) $(F_PB) --tlimit=10 --log-level=debug -r 6000
iq:
	$(KREC) $(HP) $(F_PB) -m iq --tlimit=10 --log_level=debug
iqre:
	$(KREC) $(HP) $(F_PB) -m iq --tlimit=10 --log_level=debug -r 6000
lsb:
	$(KREC) $(HP) $(F_PB) -m lsb --tlimit=10 --log-level=debug
ncomp:
	$(KREC) $(HP) $(F_PB) --ncomp
rx8:
#	$(KREC) $(H8) $(F_PB) --launch-delay=15 --socket-timeout=120 -u krec-RX8
	$(KREC) $(H8) $(F_PB) -u krec-RX8
nb:
	$(KREC) $(HP) $(F_PB) -m usb --tlimit=10 --nb --nb-gate=200 --nb-th=40
nbtest:
	$(KREC) $(HP) $(F_PB) -m usb --tlimit=10 --nb-test --nb --nb-gate=256 --nb-th=16
2sec:
	$(KREC) $(HP) $(F_PB) -q --log-level=info --dt-sec=2 
debug:
	$(KREC) $(HP) $(F_PB) --tlimit=10 --test-mode --log_level=debug
modes:
	$(KREC) $(HP) $(F_PB) -m iq  --tlimit=10 --log_level debug
	$(KREC) $(HP) $(F_PB) -m sam --tlimit=10 --log_level debug
	$(KREC) $(HP) $(F_PB) -m sau --tlimit=10 --log_level debug
	$(KREC) $(HP) $(F_PB) -m sal --tlimit=10 --log_level debug
	$(KREC) $(HP) $(F_PB) -m sas --tlimit=10 --log_level debug
	$(KREC) $(HP) $(F_PB) -m qam --tlimit=10 --log_level debug


# squelch
SQ_F_PB = -f 10000 -m cw
#SQ_F_PB = -f 4037.5 -m usb
#SQ_F_PB = -f 4788.5 -m usb

squelch sq:
	$(KREC) $(HP) $(SQ_F_PB) -T 20 --log_level=info #--test-mode


# scanning
SCAN = $(HP) --log_level=info

scan:
	$(KREC) $(SCAN) -m amn --scan-yaml=scan_continuous.yaml

scan-sm:
	$(KREC) $(SCAN) -m amn --scan-yaml=scan_continuous.yaml --S-meter=0 --sound --not-quiet

scan-sq:
	$(KREC) $(SCAN) -m usb --scan-yaml=scan_squelch.yaml


# wideband
wbr:
	@rm -f *.wav
	$(KREC) $(HP) -f 10000 -m iq --ncomp --station=wb --tlimit=30 --log_level debug --nolocal

wb:
	@rm -f *.wav
	$(KREC) $(HP) -f 10000 -m iq --ncomp --station=wb --tlimit=1000 --log_level debug --wb

wbw:
	@rm -f *.wav
	$(KREC) $(HP) -f 14035 -m iq --ncomp --station=wb --tlimit=10 --log_level debug --wb

ncwb:
	@rm -f *.wav
	$(KREC) --nc $(HP) -m iq -L 6000 -H 6000 -f 10000 --nc-wav --nc-srate 12000 --tlimit=10 --log=debug >nc.wav
#	$(KREC) --nc $(HP) -m iq -L 6000 -H 6000 -f 10000 --nc-wav --nc-srate 72000 --tlimit=10 --log=debug >nc.wav

ncw:
	@rm -f *.wav
#	$(KREC) $(HP) --nc -m iq -f 14035 --nc-wav --tlimit=10 --log=debug --wb >nc.wav

ncwp:
#	@mkfifo pipe.wav
	$(KREC) $(HP) --nc -m iq -f 14035 --nc-wav --tlimit=10 --log=debug --wb >pipe.wav


# resampling
resample:
	$(KREC) $(HP) $(F_PB) -r 6000 --tlimit=10
resample_iq:
	$(KREC) $(HP) $(F_PB) -r 6000 -m iq --tlimit=10

samplerate_build:
	@echo "See README file for complete build instructions."
	$(PY) samplerate/samplerate_build.py
samplerate_test:
	pytest --capture=tee-sys samplerate/tests/test_samplerate.py
samplerate_test2:
	$(PY) -m pytest --capture=tee-sys samplerate/tests/test_samplerate.py
info:
	sox --info *.wav


# frequency offset
FOFF = -L 470 -H 530 -m cwn --snd --wf --z 14 --speed 2 --wf-png --wf-auto --log=debug
#FOFF = -L 470 -H 530 -m cwn --snd --wf --z 14 --speed 2 --quiet --wf-png --wf-auto
#FOFF = -L 470 -H 530 -m cwn
#FOFF = -L -100 -H 100 -m iq --snd --wf --z 14 --speed 2 --quiet --wf-png --wf-auto

foff:
#	$(KREC) $(HP) --tlimit=10 --log_level=debug -f 24000.14 $(FOFF)
#	$(KREC) $(HP) --tlimit=10 -f 124000.14 $(FOFF)
	$(KREC) $(HP) --tlimit=60 -f 124000.64 -o 100000 $(FOFF)


# S-meter

# --freq-pbc means -f frequency is "passband center" freq for SSB/CW modes, not carrier/dial freq
#F_PB_SM = $(F_PB) --freq-pbc
F_PB_SM = -f 10000 --freq-pbc -m cwn
#F_PB_SM = -f 10000 --freq-pbc -m usb

s_meter sm:
	$(KREC) $(HP) $(F_PB_SM) --s-meter=10
	$(KREC) $(HP) $(F_PB_SM) --s-meter=10 -m iq
s_meter_timed smt:
	$(KREC) $(HP) $(F_PB_SM) --s-meter=10 --stats
	$(KREC) $(HP) $(F_PB_SM) --s-meter=10 --ncomp --stats
	$(KREC) $(HP) $(F_PB_SM) --s-meter=10 -m iq --stats

s_meter_stream sms:
	$(KREC) $(HP) $(F_PB_SM) --s-meter=0 --tlimit=3
#	$(KREC) $(HP) $(F_PB_SM) --s-meter=0 --tlimit=3 --stats --tstamp
#	$(KREC) $(HP) $(F_PB_SM) --s-meter=0 --tlimit=3 --stats --tstamp --sdt-sec=1
s_meter_stream_timed smst:
	$(KREC) $(HP) $(F_PB_SM) --s-meter=0 --tlimit=5 --stats
	$(KREC) $(HP) $(F_PB_SM) --s-meter=0 --tlimit=5 --ncomp --stats
	$(KREC) $(HP) $(F_PB_SM) --s-meter=0 --tlimit=5 -m iq --stats

s_meter_stream_timestamps smts:
	$(KREC) $(HP) $(F_PB_SM) --s-meter=0 --tlimit=5 --stats --tstamp

s_meter_stream_interval smsi:
	$(KREC) $(HP) $(F_PB_SM) --s-meter=0 --tstamp --sdt-sec=1 --stats --log-level=info --dt-sec=4 --snd


# TDoA debugging
H_TDOA = $(HP)
F_TDOA = -f 77.5 -m iq -L 500 -H 500 --kiwi-wav --kiwi-tdoa -u TDoA_service

tdoa:
	$(KREC) $(H_TDOA) $(F_TDOA) --tlimit=60 --log-level=debug
#	$(KREC) $(H_TDOA) $(F_TDOA) --tlimit=30 --log-level=debug --devel=0:1.005e-6,1:0


# check an iq .wav file recorded with --kiwi-wav
# make wav f=*.wav
wav:
	$(PY) kiwi/wavreader.py $(f)


# check an iq .wav file recorded with --kiwi-wav
# make proc f=*.wav
proc: oct/read_kiwi_iq_wav.oct
	octave-cli --eval "proc_kiwi_iq_wav('$(f)',255)"

install: oct/read_kiwi_iq_wav.oct

oct/%.oct: src/%.cc
	@mkdir -p oct
	cd oct && mkoctfile -I../include ../$<


# test reported problem situations

T_MODE = -m usb --ncomp     # "no compression" mode used by wsprdaemon.sh
#T_MODE = -m iq
T_PARAMS = -q --log-level=info $(HP) -u test -f 28124.6 $M -L 1200 -H 1700 --test-mode $(T_MODE)
H = $(HOST)

flood:
	while [ true ] ; do \
	    $(KREC) -s $H,$H,$H,$H,$H,$H,$H,$H -f 10000 -m usb -L 1200 -H 1700 --test-mode --tlimit=1 || true ; \
	done; true

slots1:
	$(KREC) --station=1 $(T_PARAMS) &
slots2:
	$(KREC) --station=1 $(T_PARAMS) &
	$(KREC) --station=2 $(T_PARAMS) &
slots4:
	$(KREC) --station=1 $(T_PARAMS) &
	$(KREC) --station=2 $(T_PARAMS) &
	$(KREC) --station=3 $(T_PARAMS) &
	$(KREC) --station=4 $(T_PARAMS) &
slots6:
	$(KREC) --station=1 $(T_PARAMS) &
	$(KREC) --station=2 $(T_PARAMS) &
	$(KREC) --station=3 $(T_PARAMS) &
	$(KREC) --station=4 $(T_PARAMS) &
	$(KREC) --station=5 $(T_PARAMS) &
	$(KREC) --station=6 $(T_PARAMS) &
slots8:
	$(KREC) --station=1 $(T_PARAMS) &
	$(KREC) --station=2 $(T_PARAMS) &
	$(KREC) --station=3 $(T_PARAMS) &
	$(KREC) --station=4 $(T_PARAMS) &
	$(KREC) --station=5 $(T_PARAMS) &
	$(KREC) --station=6 $(T_PARAMS) &
	$(KREC) --station=7 $(T_PARAMS) &
	$(KREC) --station=8 $(T_PARAMS) &
slots12:
	$(KREC) --station=1 $(T_PARAMS) &
	$(KREC) --station=2 $(T_PARAMS) &
	$(KREC) --station=3 $(T_PARAMS) &
	$(KREC) --station=4 $(T_PARAMS) &
	$(KREC) --station=5 $(T_PARAMS) &
	$(KREC) --station=6 $(T_PARAMS) &
	$(KREC) --station=7 $(T_PARAMS) &
	$(KREC) --station=8 $(T_PARAMS) &
	$(KREC) --station=9 $(T_PARAMS) &
	$(KREC) --station=10 $(T_PARAMS) &
	$(KREC) --station=11 $(T_PARAMS) &
	$(KREC) --station=12 $(T_PARAMS) &
slots14:
	$(KREC) --station=1 $(T_PARAMS) &
	$(KREC) --station=2 $(T_PARAMS) &
	$(KREC) --station=3 $(T_PARAMS) &
	$(KREC) --station=4 $(T_PARAMS) &
	$(KREC) --station=5 $(T_PARAMS) &
	$(KREC) --station=6 $(T_PARAMS) &
	$(KREC) --station=7 $(T_PARAMS) &
	$(KREC) --station=8 $(T_PARAMS) &
	$(KREC) --station=9 $(T_PARAMS) &
	$(KREC) --station=10 $(T_PARAMS) &
	$(KREC) --station=11 $(T_PARAMS) &
	$(KREC) --station=12 $(T_PARAMS) &
	$(KREC) --station=13 $(T_PARAMS) &
	$(KREC) --station=14 $(T_PARAMS) &

BAD = --test-mode --log_level=debug --nolocal --tlimit=30
no-api-snd:
	$(KREC) $(HP) $(F_PB) --no-api --bad-cmd --user=bad $(BAD)
no-api-snd-wf:
	$(KREC) $(HP) --snd --wf --no-api --bad-cmd --user=bad2 $(BAD)
no-api-wf:
	$(KREC) $(HP) --wf --no-api --bad-cmd --user=spaces $(BAD)
no-api-user:
	$(KREC) $(HP) --no-api --user=no_api_test $(BAD)
no-api-ext:
	$(KREC) $(HP) --no-api --bad-cmd --ext=DRM --test-mode --quiet --snd $(BAD)
no-user:
	$(KREC) $(HP) --no-api --user=none $(BAD)


# IQ file with GPS timestamps

gps:
	$(KREC) $(HP) -f 77.5 -L -5000 -H 5000 -m iq --fn=gps --kiwi-wav --log_level info --tlimit=30
gps2:
	$(KREC) $(HP) $F -m iq -L -5000 -H 5000 --kiwi-wav


# ALE 2G testing
P_ALE = -f 2784 -m usb -L 300 -H 2700 --station=ALE --resample 8000 

ale:
	$(KREC) $(HP) $(P_ALE) --log_level debug --tlimit=5
#	$(KREC) $(HP) $(P_ALE) --log_level=debug --tlimit=5 --test-mode --ext ale_2g --snd --wf --z 5


# kiwiclientd
kcd:
#	$(PY) kiwiclientd.py --help
#	$(PY) kiwiclientd.py $(HP) -f 24000 -m usb --snddev="Display Audio" --rigctl-port=6400
#	$(PY) kiwiclientd.py $(HP) -f 24000 -m usb --rigctl-port=6400 --log_level info --tlimit=5
#	$(PY) kiwiclientd.py $(HP) -f 24000 -m iq --rigctl-port=6400 --log_level info --tlimit=5 --if=200
#	$(PY) kiwiclientd.py $(HP) -f 24001.16 -m cwn --rigctl-port=6400 --log_level debug --tlimit=5 
	$(PY) kiwiclientd.py $(HP) -f 14670 -m usb --enable-rigctl --rigctl-port=6400 --log_level debug --tlimit=30
#	$(PY) kiwiclientd.py $(HP) -f 10136 -m usb --enable-rigctl --rigctl-port=6400 --log_level debug --tlimit=30
#	$(PY) kiwiclientd.py $(HP) -f 24000.7 --pbc -m am -L -500 -H 500 --log_level debug --tlimit=5 
#	$(PY) kiwiclientd.py $(HP) -f 24001.7 -m am -L -500 -H 500 --log_level debug --tlimit=5 


# time stations

BPC_HOST = $(HP)
bpc:
#	$(KREC) $(BPC_HOST) -f 68 -m iq -L 470 -H 530 --fn=BPC_cwn60_iq --tlimit=665 --log_level info
	$(KREC) $(BPC_HOST) -f 68 -m iq -L 470 -H 530 --fn=BPC_cwn60_iq --tlimit=195 --log_level info

JJY_HOST = $(HP)
jjy:
	$(KREC) $(JJY_HOST) -f 39.5 -m iq -L 470 -H 530 --fn=JJY_cwn60_iq --tlimit=195 --log_level info

RTZ_HOST = $(HP)
rtz:
	$(KREC) $(RTZ_HOST) -f 49.6 -m iq -L 485 -H 515 --fn=RTZ_cwn30_iq --tlimit=195 --log_level info

MSF_HOST = $(HP)
msf:
	$(KREC) $(MSF_HOST) -f 59.5 -m iq -L 470 -H 530 --fn=MSF_cwn60_iq --tlimit=15 --log_level info

WWVB_HOST = $(HP)
wwvb:
	$(KREC) $(WWVB_HOST) -f 59.5 -m iq -L 497 -H 503 --fn=WWVB_cwn6_iq --tlimit=195 --log_level info


# simulate SuperSDR connection
ss:
	$(KREC) $(HP) -f 15000 -m usb --user=SuperSDR-sim --log_level=debug --tlimit=10 --test-mode --wf --z 5 --snd
ssn:
	$(KREC) $(HP) -f 15000 -m usb --user=SuperSDR-sim --log_level=debug --tlimit=10 --test-mode --wf --z 5 --snd --nolocal --pw=up &
	$(KREC) $(HP) -f 15000 -m usb --user=SuperSDR-sim --log_level=debug --tlimit=10 --test-mode --wf --z 5 --snd --nolocal --pw=up &
	$(KREC) $(HP) -f 15000 -m usb --user=SuperSDR-sim --log_level=debug --tlimit=10 --test-mode --wf --z 5 --snd --nolocal --pw=up &


# process waterfall data

wf:
	$(KREC) --wf $(HP) -f 15000 -z 0 --log_level info -u krec-WF --tlimit=5
#	$(KREC) --wf $(HP) -f 5600 -z 10 --log_level info --tlimit=20 --nolocal
#	$(KREC) --wf $(HP) -f 9650 -z 4 --log_level debug -u krec-WF --tlimit=60

wf2:
	$(PY) kiwiwfrecorder.py $(HP) -f $(FREQ) -z 4 --log_level debug -u krec-WF

wf-png:
	$(KREC) --wf $(HP) -f 15000 -z 0 --log_level info -u krec-WF --tlimit=10 --wf-png --mindb=-109 --maxdb=-43

wf-peaks:
	$(KREC) --wf $(HP) -f 15000 -z 0 --log_level info -u krec-WF --tlimit=2 --mindb=-100 --maxdb=-20 --nq --speed=1 --wf-peaks=5
#	$(KREC) --wf $(HP) -f 1000 -z 4 --log_level info -u krec-WF --tlimit=5 --mindb=-100 --maxdb=-20 --nq --speed=1 -wf-peaks=5

wf-bug:
	$(KREC) $(HP) -f 15000 -m usb --tlimit=120 --snd --wf --z 0 --speed 0 --wf-comp  --wf-auto --stats --log-level=debug


# DX labels

dx-list dx:
	$(KREC) $(HP) --dx-list=9 12 --log_level=info
	cat dx.json

dx-add:
	$(KREC) $(HP) --admin --pw=apai64 -f 10 --mode=cw --dx-ident='foo bar' --dx-type=3 --dx-add --log_level=info

dx-del:
	$(KREC) $(HP) --admin --pw=apai64 --dx-del=0 --log_level=info


SNR = $(HP)

snr:
	$(KREC) --wf $(SNR) -f 15000 -z 0 --log_level=debug -u ZL4VO-snr --tlimit=15 --snr=5
#	$(KREC) --wf $(SNR) -f 10000 -z 1 --log_level=info -u ZL4VO-snr --tlimit=15 --snr=5

micro:
#	$(PY) microkiwi_waterfall.py --help
#	$(PY) microkiwi_waterfall.py $(HP)
	$(PY) microkiwi_waterfall.py $(HP) -z 0 -o 0
#	$(PY) microkiwi_waterfall.py $(HP) -z 1 -o 7500
#	$(PY) microkiwi_waterfall.py $(HP) -z 14 -o 7500
#	$(PY) microkiwi_waterfall.py $(HP) -z 0 -o 28000


# stream a Kiwi connection in a "netcat" style fashion

H_NC = $(HP)
F_NC = $(F_PB)

nc:
	$(KREC) --nc $(H_NC) $(F_NC) --tlimit=10 --log=debug --nc-wav --progress >nc.wav
#	$(KREC) --nc $(H_NC) $(F_NC) --tlimit=10 --log=info --nc-wav --progress >nc.wav
#	$(KREC)      $(H_NC) $(F_NC) --tlimit=10 --log=debug 

nciq:
	$(KREC) --nc $(H_NC) $(F_NC) -m iq --tlimit=10 --log=debug --nc-wav --progress >nciq.wav

#	$(KREC) --nc $(H_NC) $(F_PB) -m am --progress --log_level info --tlimit=3
#	$(KREC) --nc $(H_NC) -m iq -f $(HFDL_FREQ) --agc-yaml fast_agc.yaml --progress --tlimit=3 --log=debug
#	$(KREC) --nc $(H_NC) -m usb -f $(HFDL_FREQ) --progress --tlimit=25 --log=debug --nolocal
#	$(KREC) --nc $(H_NC) -m usb -f $(HFDL_FREQ) --progress --tlimit=25 --log=debug --nolocal -u foo
#	$(KREC) --nc $(H_NC) -m cw -f $(HFDL_FREQ) --progress --tlimit=3 --log=debug --wf

ncre:
	$(KREC) --nc $(H_NC) $(F_NC) --tlimit=10 --log=debug --nc-wav --progress -r 6000 >ncre.wav

nciqre:
	$(KREC) --nc $(H_NC) $(F_NC) -m iq --tlimit=10 --log=debug --nc-wav --progress -r 6000 >nciqre.wav

ncwf:
	$(KREC) --wf $(H_NC) $(F_NC) -z 0 --log_level info -u krec-WF --tlimit=5 --nq --progress --nc >ncwf.nc


# camping
CAMP = --station=$(HOST) --nc --nc-wav --log=debug --camp=0

camp:
	$(KREC) $(HP) $(CAMP) --tlimit=10 --progress >camp.wav
fdx:
	echo "SET msg_log=readline" | \
	$(KREC) $(HP) $(CAMP) --fdx --tlimit=10 --log=debug --progress >camp.wav
campre:
	$(KREC) $(HP) $(CAMP) --tlimit=10 --progress -r 6000 >campre.wav
camp20:
	$(KREC) $(HP) $(CAMP) --tlimit=20 --progress >camp.wav
camp60:
	$(KREC) $(HP) $(CAMP) --tlimit=60 --progress >camp.wav


# streaming to dumphfdl
#
# Use of an HFDL-optimized passband (e.g. "-L 300 -H 2600") is not necessary here
# since dumphfdl does its own filtering. However the Kiwi HFDL extension does have it so you
# don't have to listen to noise and interference from the opposite sideband.
HFDL_HOST = $(HP)
HFDL_FREQ = 5720

dumphfdl:
	$(KREC) --nc $(HFDL_HOST) -m iq -f $(HFDL_FREQ) --user krec:dumphfdl | \
	dumphfdl --iq-file - --sample-rate 12000 --sample-format CS16 --read-buffer-size 9600 \
	--centerfreq $(HFDL_FREQ) $(HFDL_FREQ)

dumphfdl_agc_yaml:
	$(KREC) --nc $(HFDL_HOST) -m iq -f $(HFDL_FREQ) --user krec:dumphfdl --agc-yaml fast_agc.yaml | \
	dumphfdl --iq-file - --sample-rate 12000 --sample-format CS16 --read-buffer-size 9600 \
	--centerfreq $(HFDL_FREQ) $(HFDL_FREQ)


tun:
	mkfifo /tmp/si /tmp/so
	nc -l localhost 1234 >/tmp/si </tmp/so &
	ssh -f -4 -p 1234 -L 2345:localhost:8073 root@$(HOST) sleep 600 &
	$(KREC) --nc $(HP) --log debug --admin </tmp/si >/tmp/so


# test connect/busy timeout/retries
cto:
	$(KREC) -s 1.2.3.4 -m am -f 10000 --socket-timeout 1 --connect-timeout=1 --connect-retries=5
bto:
	$(KREC) $(HP) -m am -f 10000 --busy-timeout=2 --busy-retries=3


# for copying to remote hosts
EXCLUDE_RSYNC = ".DS_Store" ".git" "__pycache__" "*.pyc" "*.wav" "*.png"
RSYNC_ARGS = -av --delete $(addprefix --exclude , $(EXCLUDE_RSYNC)) . $(RSYNC_USER)@$(HOST):$(RSYNC_DIR)/$(REPO_NAME)

REPO_NAME = kiwiclient
RSYNC_USER ?= root
RSYNC_DIR ?= /root
PORT ?= 22

ifeq ($(PORT),22)
	RSYNC = rsync
else
	RSYNC = rsync -e "ssh -p $(PORT) -l $(RSYNC_USER)"
endif

rsync_bit:
	$(RSYNC) $(RSYNC_ARGS)

version ver v:
	$(KREC) --version

help h:
	@echo HOST = $(HOST)
	@echo PORT = $(PORT)
	@echo FREQ = $(FREQ)
	@echo MODE_PB = $(MODE_PB)
	@echo
	$(KREC) --help

clean:
	-rm -f *.log *.wav *.png *.txt *.npy *.nc dx.json

clean_dist: clean
	-rm -f *.pyc */*.pyc oct/*.oct
	-rm -rf __pycache__ */__pycache__
