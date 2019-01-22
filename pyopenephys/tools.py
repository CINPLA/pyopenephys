from __future__ import division
from __future__ import print_function
from __future__ import with_statement

import quantities as pq
import os
import numpy as np


def clip_anas(analog_signals, times, clipping_times, start_end):
    '''

    Parameters
    ----------
    analog_signals
    times
    clipping_times
    start_end

    '''
    if len(analog_signals.signal) != 0:
        times = analog_signals.times.rescale(pq.s)
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
            analog_signals.signal = analog_signals.signal[:, idx[0]]
        else:
            analog_signals.signal = analog_signals.signal[idx[0]]
        analog_signals.times = times[idx]


def clip_events(events, clipping_times, start_end):
    '''

    Parameters
    ----------
    digital_signals
    clipping_times
    start_end

    Returns
    -------

    '''
    if len(events.times) != 0:
        times = events.times.rescale(pq.s)
        if len(clipping_times) == 2:
            idx = np.where((times > clipping_times[0]) & (times < clipping_times[1]))
        elif len(clipping_times) ==  1:
            if start_end == 'start':
                idx = np.where(times > clipping_times[0])
            elif start_end == 'end':
                idx = np.where(times < clipping_times[0])
        else:
            raise AttributeError('clipping_times must be of length 1 or 2')

        events.times = times[idx]
        events.channel_states = events.channel_states[idx]
        events.channels = events.channels[idx]
        events.full_words = events.full_words[idx]
        if events.metadata is not None:
            events.metadata = events.metadata[idx]


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
    if len(tracking.times) != 0:
        times = tracking.times.rescale(pq.s)
        if len(clipping_times) == 2:
            idx = np.where((times > clipping_times[0]) & (times < clipping_times[1]))
        elif len(clipping_times) ==  1:
            if start_end == 'start':
                idx = np.where(times > clipping_times[0])
            elif start_end == 'end':
                idx = np.where(times < clipping_times[0])
        else:
            raise AttributeError('clipping_times must be of length 1 or 2')

        tracking.times = times[idx]
        tracking.x = tracking.x[idx]
        tracking.y = tracking.y[idx]
        tracking.width = tracking.width[idx]
        tracking.height = tracking.height[idx]
        tracking.channels = tracking.channels[idx]
        if tracking.metadata is not None:
            tracking.metadata = tracking.metadata[idx]


def clip_spiketrains(sptr, clipping_times, start_end):
    '''

    Parameters
    ----------
    sptr
    clipping_times
    start_end

    Returns
    -------

    '''
    if len(sptr.times) != 0:
        times = sptr.times.rescale(pq.s)
        if len(clipping_times) == 2:
            idx = np.where((times > clipping_times[0]) & (times < clipping_times[1]))
        elif len(clipping_times) ==  1:
            if start_end == 'start':
                idx = np.where(times > clipping_times[0])
            elif start_end == 'end':
                idx = np.where(times < clipping_times[0])
        else:
            raise AttributeError('clipping_times must be of length 1 or 2')

        sptr.times = times[idx]
        sptr.waveforms = sptr.waveforms[idx]
        sptr.electrode_indices = sptr.electrode_indices[idx]
        sptr.cluster = sptr.cluster[idx]
        if sptr.metadata is not None:
            sptr.metadata = sptr.metadata[idx]


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