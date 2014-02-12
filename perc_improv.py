__author__ = "Ian Goodfellow"

import gc
from matplotlib import pyplot
import numpy as np
import os
import sys

from pylearn2.utils import serial

_, d1, d2 = sys.argv

pyplot.hold(True)

fs = os.listdir(d1)

improve_t = 0
improve_tm = 0
improve_both = 0
total = 0

for f in sorted(fs):
    if f in ['9', '19', '21']:
        # Skip jobs that crashed in one condition or the other
        continue
    print 'running',f
    def get_t_tm(d):
        model0 = serial.load(os.path.join(d, f, 'task_0_best.pkl'))
        monitor = model0.monitor
        channels = monitor.channels
        def read_channel(s):
            return float(channels[s].val_record[-1])
        print 'job#, orig valid, valid both, new test, old test'
        v, t = map(read_channel, ['valid_y_misclass', 'test_y_misclass'])
        note = ''
        tm = monitor.get_epochs_seen()
        gc.collect()
        return t, tm
    t1, tm1 = get_t_tm(d1)
    t2, tm2 = get_t_tm(d2)

    it = t2 < t1
    itm = tm2 < tm1
    both = it and itm
    improve_t += it
    improve_tm += itm
    improve_both += improve_both
    total += 1

print 'improves test error: ', float(improve_t) / float(total)
print 'improves training time: ', float(improve_tm) / float(total)
print 'improves both: ', float(improve_both) / float(total)

