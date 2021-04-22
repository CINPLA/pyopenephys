"""
Python library for reading OpenEphys files.

Authors: Alessio Buccino @CINPLA,
         Svenn-Arne Dragly @CINPLA,
         Milad H. Mobarhan @CINPLA,
         Mikkel E. Lepperod @CINPLA
"""
import quantities as pq
import os
import os.path as op
import numpy as np
from datetime import datetime
import locale
import struct
import platform
import xmltodict
from distutils.version import LooseVersion
from pathlib import Path
import warnings
import json

from .tools import *
from .openephys_tools import *


class AnalogSignal:
    def __init__(self, channel_ids, signal, times, gains, channel_names=None, sample_rate=None):
        self.signal = signal
        self.channel_ids = channel_ids
        self.times = times
        self.gains = gains
        self.sample_rate = sample_rate
        self.channel_names = channel_names

    def __str__(self):
        return "<OpenEphys analog signal:shape: {}>".format(
            self.signal.shape
        )


class TrackingData:
    def __init__(self, times, x, y, width, height, channels, metadata):
        self.times = times
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.channels = channels
        self.metadata = metadata

    def __str__(self):
        return "<OpenEphys tracking data: times shape: {}, positions shape: {}>".format(
            self.times.shape, self.x.shape
        )


class EventData:
    def __init__(self, times, channels, channel_states, full_words, processor, node_id, metadata=None):
        self.times = times
        self.channels = channels
        self.channel_states = channel_states
        self.full_words = full_words
        self.processor = processor
        self.node_id = node_id
        self.metadata = metadata

    def __str__(self):
        return "<OpenEphys event data>"


class MessageData:
    def __init__(self, time, channel, text):
        self.time = time
        self.channel = channel
        self.text = text

    def __str__(self):
        return "<OpenEphys message data>"


class SpikeTrain:
    def __init__(self, times, waveforms,
                 electrode_indices, cluster, metadata):
        assert len(waveforms.shape) == 3 or len(waveforms.shape) == 2
        self.times = times
        self.waveforms = waveforms
        self.electrode_indices = electrode_indices
        self.cluster = cluster
        self.metadata = metadata

    @property
    def num_spikes(self):
        """
        Alias for spike_count.
        """
        return self.waveforms.shape[0]

    @property
    def num_chans(self):
        """
        Alias for channel_count.
        """
        if len(self.waveforms.shape) == 3:
            return self.waveforms.shape[1]
        else:
            return 1

    @property
    def num_frames(self):
        """
        Alias for channel_count.
        """
        if len(self.waveforms.shape) == 3:
            return self.waveforms.shape[2]
        else:
            self.waveforms.shape[1]


class File:
    """
    Class for reading experimental data from an OpenEphys dataset.
    """

    def __init__(self, foldername):
        self._absolute_foldername = foldername
        self._path, self.relative_foldername = os.path.split(foldername)

        # figure out format
        files = [f for f in sorted(os.listdir(self._absolute_foldername))]

        if np.any([f.startswith('Continuous') for f in files]):
            self.format = 'openephys'
            cont_files = [f for f in sorted(os.listdir(self._absolute_foldername))
                          if f.startswith('Continuous')]
            exp_ids = []
            for con in cont_files:
                if len(con.split('_')) == 2:
                    exp_ids.append(1)
                else:
                    exp_ids.append(int(con.split('_')[-1][0]))
            self._experiments = []
            for id in exp_ids:
                self._experiments.append(Experiment(self._absolute_foldername, id, self))

        elif np.any([f.startswith('experiment') for f in files]):
            self.format = 'binary'
            experiments_names = [f for f in sorted(os.listdir(self._absolute_foldername))
                                 if os.path.isdir(op.join(self._absolute_foldername, f))
                                 and 'experiment' in f]
            exp_ids = [int(exp[-1]) for exp in experiments_names]
            self._experiments = []
            for (rel_path, id) in zip(experiments_names, exp_ids):
                self._experiments.append(Experiment(op.join(self._absolute_foldername, rel_path), id, self))
        else:
            raise Exception("Only 'binary' and 'openephys' format are supported by pyopenephys")

    @property
    def absolute_foldername(self):
        return self._absolute_foldername

    @property
    def path(self):
        return self._path

    @property
    def experiments(self):
        return self._experiments


class Experiment:
    def __init__(self, path, id, file):
        self.file = file
        self.id = id
        self.sig_chain = dict()
        self._absolute_foldername = path
        self._recordings = []
        self.settings = None
        self.acquisition_system = None

        if self.file.format == 'openephys':
            self._path = self._absolute_foldername
            self._read_settings(id)

            # retrieve number of recordings
            if self.acquisition_system is not None:
                if self.id == 1:
                    contFile = [f for f in os.listdir(self._absolute_foldername) if 'continuous' in f and 'CH' in f
                                and len(f.split('_')) == 2][0]
                else:
                    contFile = [f for f in os.listdir(self._absolute_foldername) if 'continuous' in f and 'CH' in f
                                and '_' + str(self.id) in f][0]
                data = loadContinuous(op.join(self._absolute_foldername, contFile))
                rec_ids = np.unique(data['recordingNumber'])
                for rec_id in rec_ids:
                    self._recordings.append(Recording(self._absolute_foldername, int(rec_id), self))
            else:
                self._recordings.append(Recording(self._absolute_foldername, int(self.id), self))

        elif self.file.format == 'binary':
            self._path = op.dirname(path)
            self._read_settings(id)
            recording_names = [f for f in os.listdir(self._absolute_foldername)
                               if os.path.isdir(op.join(self._absolute_foldername, f))
                               and 'recording' in f]

            rec_ids = [int(rec[-1]) for rec in recording_names]
            for (rel_path, id) in zip(recording_names, rec_ids):
                self._recordings.append(Recording(op.join(self._absolute_foldername, rel_path), id,
                                                  self))

    @property
    def absolute_foldername(self):
        return self._absolute_foldername

    @property
    def path(self):
        return self._path

    @property
    def datetime(self):
        return self._start_datetime

    @property
    def recordings(self):
        return self._recordings

    def _read_settings(self, id):
        print('Loading Open-Ephys: reading settings...')
        if id == 1:
            set_fname = [fname for fname in os.listdir(self._path)
                         if fname == 'settings.xml']
        else:
            set_fname = [fname for fname in os.listdir(self._path)
                         if fname.startswith('settings') and fname.endswith('.xml') and str(id) in fname]

        if not len(set_fname) == 1:
            # raise IOError('Unique settings file not found')
            print("settings.xml not found. Can't load signal chain information")
            self._set_fname = None
            self.sig_chain = None
            self.setting = None
            self.format = None
            self.nchan = None
            self._start_datetime = datetime(1970, 1, 1)
        else:
            self._set_fname = op.join(self._path, set_fname[0])
            with open(self._set_fname) as f:
                xmldata = f.read()
                self.settings = xmltodict.parse(xmldata)['SETTINGS']
            # read date in US format
            if platform.system() == 'Windows':
                locale.setlocale(locale.LC_ALL, 'english')
            elif platform.system() == 'Darwin':
                # bad hack...
                try:
                    locale.setlocale(locale.LC_ALL, 'en_US.UTF8')
                except Exception:
                    pass
            else:
                locale.setlocale(locale.LC_ALL, 'en_US.UTF8')
            self._start_datetime = datetime.strptime(self.settings['INFO']['DATE'], '%d %b %Y %H:%M:%S')
            self._channel_info = {}
            self.nchan = 0
            if isinstance(self.settings['SIGNALCHAIN'], list):
                sigchain_iter = self.settings['SIGNALCHAIN']
            else:
                sigchain_iter = [self.settings['SIGNALCHAIN']]
            for sigchain in sigchain_iter:
                if isinstance(sigchain['PROCESSOR'], list):
                    processor_iter = sigchain['PROCESSOR']
                else:
                    processor_iter = [sigchain['PROCESSOR']]
                for processor in processor_iter:
                    self.sig_chain.update({processor['@name']: processor['@NodeId']})
                    if 'CHANNEL_INFO' in processor.keys() and processor['@isSource'] == '1':
                        # recorder
                        self.acquisition_system = processor['@name'].split('/')[-1]
                        self._channel_info['gain'] = {}

                        # gain for all channels
                        gain = {ch['@number']: float(ch['@gain'])
                                for chs in processor['CHANNEL_INFO'].values()
                                for ch in chs}
                        for chan in processor['CHANNEL']:
                            if chan['SELECTIONSTATE']['@record'] == '1':
                                self.nchan += 1
                                chnum = chan['@number']
                                self._channel_info['gain'][chnum] = gain[chnum]
                    elif 'CHANNEL' in processor.keys() and processor['@isSource'] == '1':
                        # recorder
                        self._ephys = True
                        self.acquisition_system = processor['@name'].split('/')[-1]
                        self._channel_info['gain'] = {}

                        for chan in processor['CHANNEL']:
                            if chan['SELECTIONSTATE']['@record'] == '1':
                                self.nchan += 1
                                chnum = chan['@number']
                                self._channel_info['gain'][chnum] = 1

            # Check openephys format
            if self.settings['CONTROLPANEL']['@recordEngine'] == 'OPENEPHYS':
                self.format = 'openephys'
            elif self.settings['CONTROLPANEL']['@recordEngine'] == 'RAWBINARY':
                self.format = 'binary'
            else:
                self.format = None
            print('Decoding data from ', self.format, ' format')

            if self.acquisition_system is not None:
                recorded_channels = sorted([int(chan) for chan in
                                            self._channel_info['gain'].keys()])
                self._channel_info['channels'] = recorded_channels

class Recording:
    def __init__(self, path, id, experiment):
        self.experiment = experiment
        self.absolute_foldername = Path(path)
        self.format = experiment.format
        self.datetime = experiment.datetime
        self.nchan = experiment.nchan
        self.sig_chain = experiment.sig_chain
        self.id = id
        self._oebin = None

        if self.format == "binary":
            events_folders = [f for f in self.absolute_foldername.iterdir() if 'events' in f.name]
            continuous_folders = [f for f in self.absolute_foldername.iterdir() if 'continuous' in f.name]
            spikes_folders = [f for f in self.absolute_foldername.iterdir() if 'spikes' in f.name]

            self._events_folder = None
            if len(events_folders) == 1:
                self._events_folder = events_folders[0]
            elif len(events_folders) > 1:
                raise Exception("More than one events folder found!")
            self._continuous_folder = None
            if len(continuous_folders) == 1:
                self._continuous_folder = continuous_folders[0]
            elif len(continuous_folders) > 1:
                raise Exception("More than one continuous folder found!")
            self._spikes_folder = None
            if len(spikes_folders) == 1:
                self._spikes_folder = spikes_folders[0]
            elif len(spikes_folders) > 1:
                raise Exception("More than one spikes folder found!")

            if LooseVersion(self.experiment.settings['INFO']['VERSION']) >= LooseVersion('0.4.4.0'):
                oebin_files = [f for f in self.absolute_foldername.iterdir() if 'oebin' in f.name]
                if len(oebin_files) == 1:
                    print("Reading oebin file")
                    with oebin_files[0].open('r') as f:
                        self._oebin = json.load(f)
                elif len(oebin_files) == 0:
                    raise FileNotFoundError("'structre.oebin' file not found! Impossible to retrieve configuration "
                                            "information")
                else:
                    raise Exception("Multiple oebin files found. Impossible to retrieve configuration information")

        self._analog_signals_dirty = True
        self._digital_signals_dirty = True
        self._channel_groups_dirty = True
        self._spiketrains_dirty = True
        self._tracking_dirty = True
        self._events_dirty = True
        self._message_dirty = True

        self._times = None
        self._start_times = []
        self._duration = []
        self._analog_signals = []
        self._tracking_signals = []
        self._event_signals = []
        self._messages = []
        self._spiketrains = []

        self._software_start_frame = None
        self._software_sample_rate = None

        self.__dict__.update(self._read_sync_message())

    @property
    def times(self):
        if self._times is None:
            if self.experiment.acquisition_system is not None:
                self._times = self.analog_signals[0].times
            elif len(self.tracking) > 0:
                self._times = self.tracking[0].times
            else:
                self._times = None

        return self._times

    @property
    def duration(self):
        if self.experiment.acquisition_system is not None:
            self._duration = self.analog_signals[0].times[-1] - self.analog_signals[0].times[0]
            return self._duration
        if 'Sources/Tracking Port' in self.sig_chain.keys():
            self._duration = self.tracking[0].times[-1] - self.tracking[0].times[0]
            return self._duration
        else:
            self._duration = 0
            return self._duration

    @property
    def sample_rate(self):
        if self.experiment.acquisition_system is not None:
            if len(self._processor_sample_rates) == 1:
                return self._processor_sample_rates[0] * pq.Hz
            else:
                if np.all([np.isclose(self._processor_sample_rates[0], sr) for sr in self._processor_sample_rates[1:]]):
                    return self._processor_sample_rates[0] * pq.Hz
                else:
                    warnings.warn("Multiple streams with different sample rates found. To access the sample rate for "
                                  "each stream use the 'sample_rate' field of the AnalogSignal object. "
                                  "Returning maximum of the first stream")
                    return np.max(self._processor_sample_rates) * pq.Hz
        else:
            return self._software_sample_rate * pq.Hz

    @property
    def start_time(self):
        if self.experiment.acquisition_system is not None:
            if len(self._start_times) == 0:
                self._read_analog_signals()
            if len(self._start_times) == 1:
                return self._start_times[0]
            else:
                if np.all([np.isclose(self._start_times[0].magnitude, stime.magnitude)
                           for stime in self._start_times[1:]]):
                    return self._start_times[0]
                else:
                    warnings.warn("Multiple streams with different start times found. To access the sample rate for each "
                                  "stream use the 'start_time' field of the AnalogSignal object."
                                  "Returning start_time of first stream")
                    return self._start_times[0]
        else:
            return self._software_start_frame / self._software_sample_rate * pq.s

    @property
    def software_sample_rate(self):
        if self._software_sample_rate is not None:
            return self._software_sample_rate * pq.Hz
        else:
            return None

    @property
    def software_start_time(self):
        if self._software_start_frame is not None:
            return self._software_start_frame / self._software_sample_rate * pq.s
        else:
            return None

    @property
    def spiketrains(self):
        if self._spiketrains_dirty:
            self._spiketrains = []
            self._read_spiketrains()

        return self._spiketrains

    @property
    def analog_signals(self):
        if self._analog_signals_dirty:
            self._analog_signals = []
            self._read_analog_signals()

        return self._analog_signals

    @property
    def tracking(self):
        if self._tracking_dirty:
            self._tracking_signals = []
            self._read_tracking()

        return self._tracking_signals

    @property
    def events(self):
        if self._events_dirty:
            self._event_signals = []
            self._read_events()

        return self._event_signals

    @property
    def messages(self):
        if self._message_dirty:
            self._messages = []
            self._read_messages()

        return self._messages

    def _read_sync_message(self):
        info = dict()
        stimes = []

        if self.format == 'binary':
            sync_messagefile = [f for f in self.absolute_foldername.iterdir() if 'sync_messages' in f.name][0]
        elif self.format == 'openephys':
            if self.experiment.id == 1:
                sync_messagefile = self.absolute_foldername / 'messages.events'
            else:
                sync_messagefile = self.absolute_foldername / f'messages_{self.experiment.id}.events'

        with sync_messagefile.open("r") as fh:
            info['_processor_sample_rates'] = []
            info['_processor_start_frames'] = []
            info['messages'] = []
            info['_software_sample_rate'] = None
            info['_software_start_frame'] = None
            while True:
                spl = fh.readline().split()
                if not spl:
                    break
                if 'Software' in spl:
                    self.processor = False
                    stime = spl[-1].split('@')
                    hz_start = stime[-1].find('Hz')
                    sr = float(stime[-1][:hz_start])
                    info['_software_sample_rate'] = sr
                    info['_software_start_frame'] = int(stime[0])
                elif 'Processor:' in spl:
                    self.processor = True
                    stime = spl[-1].split('@')
                    hz_start = stime[-1].find('Hz')
                    stimes.append(float(stime[-1][:hz_start]))
                    sr = float(stime[-1][:hz_start])
                    info['_processor_sample_rates'].append(sr)
                    info['_processor_start_frames'].append(int(stime[0]))
                else:
                    message = {'time': int(spl[0]),
                               'message': ' '.join(spl[1:])}
                    info['messages'].append(message)

        return info

    def _read_messages(self):
        if self.format == 'binary':
            if self._events_folder is not None:
                message_folder = [f for f in self._events_folder.iterdir() if 'Message_Center' in f.name][0]
                text_groups = [f for f in message_folder.iterdir()]
                if self.format == 'binary':
                    for tg in text_groups:
                        text = np.load(tg / 'text.npy')
                        ts = np.load(tg / 'timestamps.npy')
                        channels = np.load(tg / 'channels.npy')

                        ts = ts / self.sample_rate
                        ts -= self.start_time

                        if len(text) > 0:
                            for t, time, chan in zip(text, ts, channels):
                                message_data = MessageData(
                                    time=time,
                                    channel=chan,
                                    text=t.decode("utf-8"),
                                )
                                self._messages.append(message_data)
        elif self.format == 'openephys':
            pass

        self._message_dirty = False

    def _read_events(self):
        if self.format == 'binary':
            if self._events_folder is not None:
                events = []
                processor_folders = []

                if self._oebin is not None:
                    if 'events' in self._oebin.keys():
                        events = self._oebin["events"]
                    if len(events) > 0:
                        processor_folders = []
                        for ev in events:
                            # other methods to read tracking and messages
                            if 'Tracking_Port' not in ev['folder_name'] and 'Message_Center' not in ev['folder_name']:
                                processor_folders.append((self._events_folder / ev['folder_name']).parent)

                else:
                    processor_folders = [f for f in self._events_folder.iterdir() if 'Tracking_Port' not in f.name and
                                         'Message_Center' not in f.name and not f.name.startswith('.')]

                for processor_folder in processor_folders:
                    # Read TTL groups
                    TTL_groups = [f for f in processor_folder.iterdir() if 'TTL' in f.name]
                    for ttl in TTL_groups:
                        full_words = np.load(ttl / 'full_words.npy')
                        ts = np.load(ttl / 'timestamps.npy')
                        channels = np.load(ttl / 'channels.npy').astype(int)
                        unique_channels = np.unique(channels)
                        channel_states = np.load(ttl / 'channel_states.npy')
                        metadata_file = ttl / 'metadata.npy'
                        if metadata_file.is_file():
                            metadata = np.load(metadata_file)
                        else:
                            metadata = None

                        for chan in unique_channels:
                            chan_idx = np.where(channels == chan)
                            chans = channels[chan_idx]
                            fw_chans = full_words[chan_idx]
                            if metadata is not None:
                                metadata_chan = metadata[chan_idx]
                            else:
                                metadata_chan = None
                            ts_chans = ts[chan_idx]
                            if len(ts_chans) > 0:
                                chan_states = channel_states[chan_idx] / np.max(channel_states[chan_idx]).astype(int)
                            else:
                                chan_states = None

                            ts_chans = ts_chans / self.sample_rate
                            ts_chans -= self.start_time
                            processor_folder_split = processor_folder.name.split("-")

                            if len(ts) > 0:
                                event_data = EventData(
                                    times=ts_chans,
                                    channels=chans,
                                    channel_states=chan_states,
                                    full_words=fw_chans,
                                    processor=processor_folder_split[0],
                                    node_id=int(float(processor_folder_split[1])),
                                    metadata=metadata_chan
                                )
                                self._event_signals.append(event_data)

                    # Read Binary groups
                    binary_groups = [f for f in processor_folder.iterdir() if 'binary' in f.name]
                    for bg in binary_groups:
                        full_words = np.load(bg / 'full_words.npy')
                        ts = np.load(bg / 'timestamps.npy')
                        channels = np.load(bg / 'channels.npy').astype(int)
                        channel_states = np.load(bg / 'channel_states.npy')
                        channel_states = channel_states / np.max(channel_states).astype(int)
                        metadata_file = bg / 'metadata.npy'
                        if metadata_file.is_file():
                            metadata = np.load(metadata_file)
                        else:
                            metadata = None

                        if self.software_sample_rate is not None:
                            sample_rate = self.software_sample_rate
                        else:
                            sample_rate = self.sample_rate

                        ts = ts / sample_rate
                        ts -= self.start_time

                        processor_folder_split = processor_folder.name.split("-")

                        if len(ts) > 0:
                            event_data = EventData(
                                times=ts,
                                channels=channels,
                                channel_states=channel_states,
                                full_words=full_words,
                                processor=processor_folder_split[0],
                                node_id=int(float(processor_folder_split[1])),
                                metadata=metadata
                            )
                            self._event_signals.append(event_data)

        elif self.format == 'openephys':
            if self.experiment.id == 1:
                ev_file = op.join(self.absolute_foldername, 'all_channels.events')
            else:
                ev_file = op.join(self.absolute_foldername, 'all_channels_' + str(int(self.experiment.id)) + '.events')
            data = loadEvents(ev_file)
            node_ids = np.unique(data['nodeId']).astype(int)

            for node in node_ids:
                idx_ev = np.where(data['nodeId'] == node)[0]
                ts = data['timestamps'][idx_ev] / self.software_sample_rate
                channels = data['channel'][idx_ev].astype(int)
                channel_states = data['eventId'][idx_ev].astype(int)
                channel_states[channel_states == 0] = -1
                node_id = int(float(node))
                full_words = None
                metadata = None
                ts -= self.start_time

                event_data = EventData(
                    times=ts,
                    channels=channels,
                    channel_states=channel_states,
                    full_words=full_words,
                    processor=None,
                    node_id=node_id,
                    metadata=metadata
                )

                self._event_signals.append(event_data)

        self._events_dirty = False

    def _read_tracking(self):
        if 'Sources/Tracking Port' in self.sig_chain.keys():
            if self.format == 'binary':
                # Check and decode files
                if self._events_folder is not None:
                    tracking_folder = [f for f in self._events_folder.iterdir() if 'Tracking_Port' in f.name][0]
                    binary_groups = [f for f in tracking_folder.iterdir()]
                    for bg in binary_groups:
                        data_array = np.load(bg / 'data_array.npy')
                        ts = np.load(bg / 'timestamps.npy')
                        channels = np.load(bg / 'channels.npy')
                        metadata = np.load(bg / 'metadata.npy')
                        data_array = np.array([struct.unpack('4f', d) for d in data_array])

                        if self.software_sample_rate is not None:
                            sample_rate = self.software_sample_rate
                        else:
                            sample_rate = self.sample_rate

                        ts = ts / sample_rate
                        ts -= self.start_time

                        if len(ts) > 0:
                            x, y, w, h = data_array[:, 0], data_array[:, 1], data_array[:, 2], data_array[:, 3]
                            tracking_data = TrackingData(
                                times=ts,
                                x=x,
                                y=y,
                                channels=channels,
                                metadata=metadata,
                                width=w,
                                height=h
                            )
                            self._tracking_signals.append(tracking_data)

            elif self.format == 'openephys':
                print("Unfortunately, tracking is not saved in 'openephys' format. Use 'binary' instead!")
        else:
            print("Tracking is not found!")

        self._tracking_dirty = False

    def _read_analog_signals(self):
        self._analog_signals = []
        if self.experiment.acquisition_system is not None:
            if self.format == 'binary':
                # Check and decode files
                if self._continuous_folder is not None:
                    # Fix THIS!
                    if self._oebin is not None:
                        continuous = self._oebin["continuous"]

                        for cont in continuous:
                            data_folder = self._continuous_folder / cont["folder_name"]
                            nchan = cont["num_channels"]
                            sample_rate = cont["sample_rate"]
                            datfiles = [f for f in data_folder.iterdir() if f.suffix == '.dat']

                            if len(datfiles) == 1:
                                datfile = datfiles[0]
                                with datfile.open("rb") as fh:
                                    anas, nsamples = read_analog_binary_signals(fh, nchan)
                                ts = np.load(data_folder / 'timestamps.npy') / sample_rate
                                self._start_times.append(ts[0] * pq.s)
                                if len(ts) != nsamples:
                                    warnings.warn('timestamps and nsamples are different!')
                                    ts = np.arange(nsamples) / sample_rate
                                else:
                                    ts -= ts[0]

                                # retrieve channel ids and gain
                                channel_names = []
                                gains = []
                                for ch in cont["channels"]:
                                    channel_names.append(ch["channel_name"])
                                    gains.append(ch["bit_volts"])

                                ts = ts * pq.s
                                self._analog_signals.append(AnalogSignal(
                                    channel_ids=range(anas.shape[0]),
                                    channel_names=channel_names,
                                    signal=anas,
                                    times=ts,
                                    gains=gains,
                                    sample_rate=sample_rate
                                ))
                            else:
                                raise ValueError("'continuous.dat' should be in the folder")
                    else:
                        fixed_gain = 0.195
                        processor_folders = [f for f in self._continuous_folder.iterdir() if f.is_dir()]
                        if len(processor_folders) > 1:
                            for c in processor_folders:
                                # only get source continuous processors
                                if 'Rhythm_FPGA' in c or 'Intan' in c or 'File' in c or "NPIX" in c:
                                    processor_folder = c
                        else:
                            processor_folder = processor_folders[0]
                        filenames = [f for f in os.listdir(processor_folder)]

                        if any('.dat' in f for f in filenames):
                            datfile = [f for f in filenames if '.dat' in f and 'continuous' in f][0]
                            with open(op.join(processor_folder, datfile), "rb") as fh:
                                anas, nsamples = read_analog_binary_signals(fh, self.nchan)
                            ts = np.load(op.join(processor_folder, 'timestamps.npy')) / self.sample_rate
                            self._start_times.append(ts[0] * pq.s)
                            if len(ts) != nsamples:
                                warnings.warn('timestamps and nsamples are different!')
                                ts = np.arange(nsamples) / self.sample_rate.magnitude
                            else:
                                ts -= ts[0]
                        else:
                            raise ValueError("'continuous.dat' should be in the folder")

                        ts = ts * pq.s
                        self._analog_signals.append(AnalogSignal(
                            channel_ids=range(anas.shape[0]),
                            signal=anas,
                            times=ts,
                            sample_rate=self.sample_rate.magnitude,
                            gains=np.ones(anas.shape[0]) * fixed_gain
                        ))

            elif self.format == 'openephys':
                fixed_gain = 0.195
                # Find continuous CH data
                if self.experiment.id == 1:
                    cont_files = [f for f in self.absolute_foldername.iterdir() if 'continuous' in f.name and
                                  'CH' in f.name and len(f.name.split('_')) == 2]
                else:
                    cont_files = [f for f in self.absolute_foldername.iterdir() if 'continuous' in f.name and
                                  'CH' in f.name and '_' + str(self.experiment.id) in f.name]

                # sort channels
                idxs = [int(x.name[x.name.find('CH') + 2: x.name.find('.')]) for x in cont_files]
                cont_files = list(np.array(cont_files)[np.argsort(idxs)])

                if len(cont_files) > 0:
                    anas = np.array([])
                    sample_rate = None
                    for i_f, f in enumerate(cont_files):
                        fullpath = f
                        sig = loadContinuous(str(fullpath))
                        block_len = int(sig['header']['blockLength'])
                        sample_rate = float(sig['header']['sampleRate'])
                        if anas.shape[0] < 1:
                            anas = sig['data'][None, :]
                        else:
                            if sig['data'].size == anas[-1].size:
                                anas = np.append(anas, sig['data'][None, :], axis=0)
                            else:
                                raise Exception('Channels must have the same number of samples')

                        if i_f == len(cont_files) - 1:
                            # Recordings number
                            rec_num = sig['recordingNumber']
                            timestamps = sig['timestamps']
                            idx_rec = np.where(rec_num == self.id)[0]
                            if len(idx_rec) > 0:
                                idx_start = idx_rec[0]
                                idx_end = idx_rec[-1]
                                t_start = timestamps[idx_start]
                                t_end = timestamps[idx_end] + block_len
                                anas_start = idx_start * block_len
                                anas_end = (idx_end + 1) * block_len
                                ts = np.arange(t_start, t_end) / sample_rate * pq.s
                                self._start_times.append(ts[0])
                                ts -= ts[0]
                                anas = anas[:, anas_start:anas_end]

                    self._processor_sample_rates = [sample_rate]
                    self._analog_signals = [AnalogSignal(
                        channel_ids=range(anas.shape[0]),
                        signal=anas,
                        times=ts,
                        sample_rate=self.sample_rate.magnitude,
                        gains=np.ones(anas.shape[0]) * fixed_gain,
                    )]
        else:
            self._analog_signals = [AnalogSignal(
                channel_ids=np.array([]),
                signal=np.array([]),
                times=np.array([]),
                gains=0
            )]

        self._analog_signals_dirty = False

    def _read_spiketrains(self):
        if self.format == 'binary':
            # Check and decode files
            if self._spikes_folder is not None:
                processor_folders = [f for f in self._spikes_folder.iterdir() if f.is_dir() and
                                     not f.name.startswith('.')]

                for processor_folder in processor_folders:
                    spike_groups = [f for f in processor_folder.iterdir() if not f.name.startswith('.')]
                    for sg in spike_groups:
                        spike_clusters = np.load(sg / 'spike_clusters.npy')
                        spike_electrode_indices = np.load(sg / 'spike_electrode_indices.npy')
                        spike_times = np.load(sg / 'spike_times.npy')
                        spike_waveforms = np.load(sg / 'spike_waveforms.npy')

                        metadata_file = sg / 'metadata.npy'
                        if metadata_file.is_file():
                            metadata = np.load(metadata_file)
                        else:
                            metadata = None

                        spike_times = spike_times / self.sample_rate
                        spike_times -= self.start_time
                        clusters = np.unique(spike_clusters)
                        for clust in clusters:
                            idx = np.where(spike_clusters == clust)[0]
                            spiketrain = SpikeTrain(times=spike_times[idx],
                                                    waveforms=spike_waveforms[idx],
                                                    electrode_indices=spike_electrode_indices[idx],
                                                    cluster=clust,
                                                    metadata=metadata)
                            self._spiketrains.append(spiketrain)

        elif self.format == 'openephys':
            filenames = [f for f in self.absolute_foldername.iterdir() if f.suffix == '.spikes']
            # order channels
            idxs = [int(x.name.split('.')[1][x.name.split('.')[1].find('0n') + 2:]) for x in filenames]
            filenames = list(np.array(filenames)[np.argsort(idxs)])

            if len(filenames) != 0:
                self._spiketrains = []
                spike_clusters = np.array([])
                spike_times = np.array([])
                spike_electrode_indices = np.array([])
                spike_waveforms = np.array([])
                if len(filenames) == 0:
                    return
                for i_f, fpath in enumerate(filenames):
                    fname = fpath.name
                    data = loadSpikes(str(fpath))

                    if i_f == 0:
                        spike_clusters = np.max(data['sortedId'], axis=1).astype(int)
                        spike_times = data['timestamps']
                        spike_electrode_indices = np.array([int(fname[fname.find('0n') + 2]) + 1]
                                                           * len(spike_clusters))
                        spike_waveforms = data['spikes'].swapaxes(1, 2)
                    else:
                        spike_clusters = np.hstack((spike_clusters, np.max(data['sortedId'], axis=1).astype(int)))
                        spike_times = np.hstack((spike_times, data['timestamps']))
                        spike_electrode_indices = np.hstack((spike_electrode_indices,
                                                             np.array([int(fname[fname.find('0n') + 2]) + 1]
                                                                      * len(data['sortedId']))))
                        spike_waveforms = np.vstack((spike_waveforms, data['spikes'].swapaxes(1, 2)))

                clusters = np.unique(spike_clusters)
                spike_times = spike_times / self.sample_rate
                spike_times -= self.start_time
                for clust in clusters:
                    idx = np.where(spike_clusters == clust)[0]
                    spiketrain = SpikeTrain(times=spike_times[idx],
                                            waveforms=spike_waveforms[idx],
                                            electrode_indices=spike_electrode_indices[idx],
                                            cluster=clust,
                                            metadata=None)
                    self._spiketrains.append(spiketrain)

        self._spiketrains_dirty = False

    def clip_recording(self, clipping_times, start_end='start'):
        """
        Clips recording, including analog signals, events, spike trains, and tracking

        Parameters
        ----------
        clipping_times: float
            Clipping times. It can have 1 or 2 elements. Assumed in seconds
        start_end: str
            'start' or 'end'. If len(clipping_times) is 1, whether to use it as start or end time.
        """
        if isinstance(clipping_times, pq.quantity.Quantity):
            raise Exception("Please provide the 'clipping_times' in float (seconds)")
        if clipping_times is not None:
            if not isinstance(clipping_times, (list, np.ndarray)):
                clipping_times = [clipping_times]

            clipping_times_pq = []
            for clip in clipping_times:
                if not isinstance(clip, pq.quantity.Quantity):
                    clipping_times_pq.append(clip * pq.s)
                else:
                    clipping_times_pq.append(clip)
            clipping_times_pq = [t.rescale(pq.s) for t in clipping_times_pq]

            if np.any([self.times[0] < clip_t < self.times[-1]] for clip_t in clipping_times_pq):
                times = self.times
                for anas in self.analog_signals:
                    clip_anas(anas, clipping_times_pq, start_end)
                for ev in self.events:
                    clip_events(ev, clipping_times_pq, start_end)
                for track in self.tracking:
                    clip_tracking(track, clipping_times_pq, start_end)
                for sptr in self.spiketrains:
                    clip_spiketrains(sptr, clipping_times_pq, start_end)

                self._times = clip_times(times, clipping_times_pq, start_end)
                self._duration = self._times[-1] - self._times[0]
            else:
                print("Clipping times are outside of timestamps range")
        else:
            print('Empty clipping times list.')

    def export_matlab(self, filename):
        from scipy import io as sio

        dict_to_save = {'duration': self.duration.rescale('s'), 'timestamps': self.times.rescale('s')}

        if len(self.tracking) != 0:
            dict_to_save.update({'tracking': np.array([[tr.x, tr.y] for tr in self.tracking])})
        if len(self.analog_signals) != 0:
            dict_to_save.update({'analog': np.array([sig.signal for sig in self.analog_signals])})
        if len(self.events) != 0:
            dict_to_save.update({'events': np.array([ev.times for ev in self.events])})

        sio.savemat(filename, dict_to_save)
