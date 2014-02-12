__author__ = "Ian Goodfellow"

import gc
from matplotlib import pyplot
import numpy as np
import os
import sys

from pylearn2.utils import serial

_, d1, d2 = sys.argv

pyplot.hold(True)

for i, d in enumerate([d1, d2]):
    fs = os.listdir(d)

    best = np.inf

    x = []
    y = []

    assert len(fs) > 0

    for f in sorted(fs):
        if f in ['9', '19', '21']:
            # Skip jobs that crashed in one condition or the other
            continue
        print 'running',f
        model = serial.load(os.path.join(d, f, 'task_0_best.pkl'))
        monitor = model.monitor
        channels = monitor.channels
        def read_channel(s):
            return float(channels[s].val_record[-1])
        print 'job#, orig valid, valid both, new test, old test'
        v, t = map(read_channel, ['valid_y_misclass', 'test_y_misclass'])
        note = ''
        if v < best:
            note = '!'
            best = v
        tm = monitor.get_epochs_seen()
        print ' '.join([note, str(f), str(v), str(t), str(tm)])

        x.append(t)
        y.append(tm)

        gc.collect()

    assert len(x) > 0
    assert len(y) > 0
    pyplot.scatter(x, y, marker = 'ox'[i], label=['no momentum', 'momentum'][i])

# pyplot.gca().set_xscale('log')
# pyplot.gca().set_yscale('log')
pyplot.xlabel('Test set error rate')
pyplot.ylabel('Training epochs')
pyplot.legend()
pyplot.show()
