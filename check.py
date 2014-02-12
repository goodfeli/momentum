__author__ = "Ian Goodfellow"

import gc
import numpy as np
import os
import sys

from pylearn2.utils import serial

_, d = sys.argv

fs = os.listdir(d)

best = np.inf

for f in fs:
    try:
        model = serial.load(os.path.join(d, f, 'task_0_best.pkl'))
    except Exception, e:
        print "trouble with ", f
        continue
    monitor = model.monitor
    channels = monitor.channels
    def read_channel(s):
        return float(channels[s].val_record[-1])
    v, t = map(read_channel, ['valid_y_misclass', 'test_y_misclass'])
    note = ''
    if v < best:
        note = '!'
        best = v
    print ' '.join([note, str(f), str(v), str(t), str(monitor.get_epochs_seen())])
    gc.collect()
