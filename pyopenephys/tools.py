from __future__ import division
from __future__ import print_function
from __future__ import with_statement

import quantities as pq
import os
import os.path as op
import numpy as np


def read_python(path):
    from six import exec_
    path = op.realpath(op.expanduser(path))
    assert op.exists(path)
    with open(path, 'r') as f:
        contents = f.read()
    metadata = {}
    exec_(contents, {}, metadata)
    metadata = {k.lower(): v for (k, v) in metadata.items()}
    return metadata


# TODO require quantities and deal with it
def clip_anas(analog_signals, times, clipping_times, start_end):
    '''

    Parameters
    ----------
    analog_signals
    times
    clipping_times
    start_end

    Returns
    -------

    '''

    if len(analog_signals.signal) != 0:
        times.rescale(pq.s)
        if len(clipping_times) == 2:
            idx = np.where((times > clipping_times[0]) & (times < clipping_times[1]))
        elif len(clipping_times) ==  1:
            if start_end == 'start':
                idx = np.where(times > clipping_times[0])
            elif start_end == 'end':
                idx = np.where(times < clipping_times[0])
        else:
            raise AttributeError('clipping_times must be of length 1 or 2')

        if len(analog_signals.signal.shape) == 2:
            anas_clip = analog_signals.signal[:, idx[0]]
        else:
            anas_clip = analog_signals.signal[idx[0]]

        return anas_clip
    else:
        return []


def clip_events(digital_signals, clipping_times, start_end):
    pass


def clip_tracking(tracking, clipping_times, start_end):
    '''

    Parameters
    ----------
    tracking
    clipping_times
    start_end

    Returns
    -------

    '''
    assert len(tracking.positions) == len(tracking.times)

    track_clip = []
    t_clip = []

    for i, tr in enumerate(tracking.positions):
        tracking.times[i].rescale(pq.s)
        if len(clipping_times) == 2:
            idx = np.where((tracking.times[i] > clipping_times[0]) & (tracking.times[i] < clipping_times[1]))
        elif len(clipping_times) ==  1:
            if start_end == 'start':
                idx = np.where(tracking.times[i] > clipping_times[0])
            elif start_end == 'end':
                idx = np.where(tracking.times[i] < clipping_times[0])
        else:
            raise AttributeError('clipping_times must be of length 1 or 2')

        track_clip.append(np.array([led[idx[0]] for led in tr]))
        if start_end != 'end':
            times = tracking.times[i][idx[0]] - clipping_times[0]
        else:
            times = tracking.times[i][idx[0]]
        t_clip.append(times)

    return track_clip, t_clip


def clip_spiketrains(tracking, clipping_times, start_end):
    pass


def clip_times(times, clipping_times, start_end='start'):
    '''

    Parameters
    ----------
    times
    clipping_times
    start_end

    Returns
    -------

    '''
    times.rescale(pq.s)

    if len(clipping_times) == 2:
        idx = np.where((times > clipping_times[0]) & (times <= clipping_times[1]))
    elif len(clipping_times) ==  1:
        if start_end == 'start':
            idx = np.where(times >= clipping_times[0])
        elif start_end == 'end':
            idx = np.where(times <= clipping_times[0])
    else:
        raise AttributeError('clipping_times must be of length 1 or 2')
    times_clip = times[idx]

    return times_clip


def read_analog_binary_signals(filehandle, numchan):

    numchan=int(numchan)

    nsamples = os.fstat(filehandle.fileno()).st_size // (numchan*2)
    print('Estimated samples: ', int(nsamples), ' Numchan: ', numchan)

    samples = np.memmap(filehandle, np.dtype('i2'), mode='r',
                        shape=(nsamples, numchan))
    samples = np.transpose(samples)

    return samples, nsamples