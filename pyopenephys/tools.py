import quantities as pq
import os
import numpy as np


def clip_anas(analog_signals, clipping_times, start_end='start'):
    """
    Clips analog signals

    Parameters
    ----------
    analog_signals: list
        List of AnalogSignal objects
    clipping_times: list
        List with clipping times. It can have 1 or 2 elements
    start_end: str
        'start' or 'end'. If len(clipping_times) is 1, whether to use it as start or end time.

    """
    if len(analog_signals.signal) != 0:
        print(analog_signals.times)
        times = analog_signals.times.rescale(pq.s)
        if len(clipping_times) == 2:
            idx = np.where((times > clipping_times[0]) & (times < clipping_times[1]))
        elif len(clipping_times) == 1:
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
    """
    Clips event signals

    Parameters
    ----------
    events: list
        List of Event objects
    clipping_times: list
        List with clipping times. It can have 1 or 2 elements
    start_end: str
        'start' or 'end'. If len(clipping_times) is 1, whether to use it as start or end time.
    """
    if len(events.times) != 0:
        times = events.times.rescale(pq.s)
        if len(clipping_times) == 2:
            idx = np.where((times > clipping_times[0]) & (times < clipping_times[1]))
        elif len(clipping_times) == 1:
            if start_end == 'start':
                idx = np.where(times > clipping_times[0])
            elif start_end == 'end':
                idx = np.where(times < clipping_times[0])
        else:
            raise AttributeError('clipping_times must be of length 1 or 2')

        events.times = times[idx]
        events.channel_states = events.channel_states[idx]
        events.channels = events.channels[idx]
        if events.full_words is not None:
            events.full_words = events.full_words[idx]
        if events.metadata is not None:
            events.metadata = events.metadata[idx]


def clip_tracking(tracking, clipping_times, start_end):
    """
    Clips tracking signals

    Parameters
    ----------
    tracking: list
        List of Tracking objects
    clipping_times: list
        List with clipping times. It can have 1 or 2 elements
    start_end: str
        'start' or 'end'. If len(clipping_times) is 1, whether to use it as start or end time.
    """
    if len(tracking.times) != 0:
        times = tracking.times.rescale(pq.s)
        if len(clipping_times) == 2:
            idx = np.where((times > clipping_times[0]) & (times < clipping_times[1]))
        elif len(clipping_times) == 1:
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
    """
    Clips spike trains

    Parameters
    ----------
    sptr: list
        List of Tracking objects
    clipping_times: list
        List with clipping times. It can have 1 or 2 elements
    start_end: str
        'start' or 'end'. If len(clipping_times) is 1, whether to use it as start or end time.
    """
    if len(sptr.times) != 0:
        times = sptr.times.rescale(pq.s)
        if len(clipping_times) == 2:
            idx = np.where((times > clipping_times[0]) & (times < clipping_times[1]))
        elif len(clipping_times) == 1:
            if start_end == 'start':
                idx = np.where(times > clipping_times[0])
            elif start_end == 'end':
                idx = np.where(times < clipping_times[0])
        else:
            raise AttributeError('clipping_times must be of length 1 or 2')

        sptr.times = times[idx]
        sptr.waveforms = sptr.waveforms[idx]
        sptr.electrode_indices = sptr.electrode_indices[idx]
        if sptr.metadata is not None:
            sptr.metadata = sptr.metadata[idx]


def clip_times(times, clipping_times, start_end='start'):
    """
    Clips timestamps

    Parameters
    ----------
    times: quantity array
        The timestamps
    clipping_times: list
        List with clipping times. It can have 1 or 2 elements
    start_end: str
        'start' or 'end'. If len(clipping_times) is 1, whether to use it as start or end time.
    """
    times.rescale(pq.s)

    if len(clipping_times) == 2:
        idx = np.where((times > clipping_times[0]) & (times <= clipping_times[1]))
    elif len(clipping_times) == 1:
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
    samples = np.memmap(filehandle, np.dtype('i2'), mode='r',
                        shape=(nsamples, numchan))
    samples = np.transpose(samples)

    return samples, nsamples

