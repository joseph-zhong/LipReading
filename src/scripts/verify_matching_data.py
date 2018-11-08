#!/usr/bin/env python3
import glob
import os

mp4_glob = 'data/raw/StephenColbert/all/*.mp4'
for mp4 in sorted(glob.glob(mp4_glob)):
    basename = mp4.split('.')[0]
    if not os.path.isfile(basename + '.en.vtt'):
        print(mp4, basename + '.en.vtt')


