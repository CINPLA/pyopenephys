"""
Python library for reading OpenEphys files.
Depends on: sys
            os
            glob
            datetime
            numpy
            quantities
            xmljson
            xmltodict

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
from .tools import *
from .openephys_tools import *


class Channel:
    def __init__(self, index, name, gain, channel_id):
        self.index = index
        self.id = channel_id
        self.name = name
        self.gain = gain


class AnalogSignal:
    def __init__(self, channel_id, signal, times):
        self.signal = signal
        self.channel_id = channel_id
        self.times = times

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
    def __init__(self, times, channels, text):
        self.times = times
        self.channels = channels
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


#todo fix channels where they belong!
class ChannelGroup:
    def __init__(self, channel_group_id, filename, channels,
                 fileclass=None, **attrs):
        self.attrs = attrs
        self.filename = filename
        self.id = channel_group_id
        self.channels = channels
        self.fileclass = fileclass

    def __str__(self):
        return "<OpenEphys channel_group {}: channel_count: {}>".format(
            self.id, len(self.channels)
        )

    @property
    def analog_signals(self):
        ana = self.fileclass.analog_signals[0]
        analog_signals = []
        for channel in self.channels:
            analog_signals.append(AnalogSignal(signal=ana.signal[channel.id],
                                               channel_id=channel.id,
                                               times=ana.times))
        return analog_signals

    @property
    def spiketrains(self):
        return [sptr for sptr in self.fileclass.spiketrains
                if sptr.attrs['channel_group_id'] == self.id]


class File:
    """
    Class for reading experimental data from an OpenEphys dataset.
    """
    def __init__(self, foldername, probefile=None):
        # TODO assert probefile is a probefile
        # TODO add default prb map and allow to add it later
        self.probefile = probefile
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
        elif np.any([f.endswith('nwb') for f in files]):
            self.format = 'nwb'


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
        self.probefile = file.probefile
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
        print('Loading Open-Ephys: reading settings.xml...')
        if id == 1:
            set_fname = [fname for fname in os.listdir(self._path)
                         if fname == 'settings.xml']
        else:
            set_fname = [fname for fname in os.listdir(self._path)
                         if fname.startswith('settings') and fname.endswith('.xml') and str(id) in fname]

        if not len(set_fname) == 1:
            raise IOError('Unique settings file not found')

        self._set_fname = op.join(self._path, set_fname[0])
        with open(self._set_fname) as f:
            xmldata = f.read()
            self.settings  = xmltodict.parse(xmldata)['SETTINGS']
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
                    gain = {ch['@number']: float(ch['@gain']) * pq.uV  # TODO assert is uV
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
            if self.probefile is not None:
                self._keep_channels = []
                self.probefile_ch_mapping = read_python(self.probefile)['channel_groups']
                for group_idx, group in self.probefile_ch_mapping.items():
                    group['gain'] = []
                    # prb file channels are sequential, 'channels' are not as they depend on FPGA channel selection
                    # -> Collapse them into array
                    for chan, oe_chan in zip(group['channels'],
                                             group['oe_channels']):
                        if oe_chan not in recorded_channels:
                            raise ValueError('Channel "' + str(oe_chan) +
                                             '" in channel group "' +
                                             str(group_idx) + '" in probefile' +
                                             self.probefile +
                                             ' is not marked as recorded ' +
                                             'in settings file' +
                                             self._set_fname)
                        group['gain'].append(
                            self._channel_info['gain'][str(oe_chan)]
                        )
                        self._keep_channels.append(recorded_channels.index(oe_chan))
                print('Number of selected channels: ', len(self._keep_channels))
            else:
                self.probefile_ch_mapping = None
                self._keep_channels = None  # HACK
        else:
            self.probefile_ch_mapping = None
            self._keep_channels = None


class Recording:
    def __init__(self, path, id, experiment):
        self.experiment = experiment
        self.absolute_foldername = path
        self.probefile = experiment.probefile
        self.sig_chain = experiment.sig_chain
        self.format = experiment.format
        self._keep_channels = experiment._keep_channels
        self.nchan = experiment.nchan
        self.probefile_ch_mapping = experiment.probefile_ch_mapping
        self.id = id

        self._analog_signals_dirty = True
        self._digital_signals_dirty = True
        self._channel_groups_dirty = True
        self._spiketrains_dirty = True
        self._tracking_dirty = True
        self._events_dirty = True
        self._message_dirty = True

        self._times = []
        self._duration = []
        self._analog_signals = []
        self._tracking_signals = []
        self._event_signals = []
        self._messages = []
        self._spiketrains = []

        self.__dict__.update(self._read_sync_message())


    @property
    def times(self):
        if self.experiment.acquisition_system is not None:
            if not self._analog_signals_dirty and self.nchan != 0:
                self._times = self.analog_signals[0].times
        if 'Sources/Tracking Port' in self.sig_chain.keys():
            self._times = self.tracking[0].times
        else:
            self._times = []

        return self._times

    @property
    def duration(self):
        if self.experiment.acquisition_system is not None:
            # if not self._analog_signals_dirty and self.nchan != 0:
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
            return self._processor_sample_rate
        else:
            return self._software_sample_rate

    @property
    def start_time(self):
        if self.experiment.acquisition_system is not None:
            return self._processor_start_time / self.sample_rate
        else:
            return self._software_start_time / self.sample_rate

    @property
    def software_sample_rate(self):
        return self._software_sample_rate

    # TODO pass channel info from exp
    def channel_group(self, channel_id):
        if self._channel_groups_dirty:
            self._read_channel_groups()
        return self._channel_id_to_channel_group[channel_id]

    @property
    def channel_groups(self):
        if self._channel_groups_dirty:
            self._read_channel_groups()

        return self._channel_groups

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
            sync_messagefile = [f for f in os.listdir(self.absolute_foldername) if 'sync_messages' in f][0]
        elif self.format == 'openephys':
            if self.experiment.id == 1:
                sync_messagefile = 'messages.events'
            else:
                sync_messagefile = 'messages_' + str(self.experiment.id) + '.events'

        with open(op.join(self.absolute_foldername, sync_messagefile), "r") as fh:
            while True:
                spl = fh.readline().split()
                if not spl:
                    break
                if 'Software' in spl:
                    self.processor = False
                    stime = spl[-1].split('@')
                    hz_start = stime[-1].find('Hz')
                    sr = float(stime[-1][:hz_start]) * pq.Hz
                    info['_software_sample_rate'] = sr
                    info['_software_start_time'] = int(stime[0])
                elif 'Processor:' in spl:
                    self.processor = True
                    stime = spl[-1].split('@')
                    hz_start = stime[-1].find('Hz')
                    stimes.append(float(stime[-1][:hz_start]))
                    sr = float(stime[-1][:hz_start]) * pq.Hz
                    info['_processor_sample_rate'] = sr
                    info['_processor_start_time'] = int(stime[0])
                else:
                    message = {'time': int(spl[0]),
                               'message': ' '.join(spl[1:])}
                    info['messages'].append(message)
        if any(np.diff(np.array(stimes, dtype=int))):
            raise ValueError('Found different processor start times')

        return info


    def _read_messages(self):
        if self.format == 'binary':
            events_folder = [op.join(self.absolute_foldername, f)
                             for f in os.listdir(self.absolute_foldername) if 'events' in f][0]
            message_folder = [op.join(events_folder, f) for f in os.listdir(events_folder)
                               if 'Message_Center' in f][0]
            text_groups = [f for f in os.listdir(message_folder)]
            if self.format == 'binary':
                for tg in text_groups:
                    text = np.load(op.join(message_folder, tg, 'text.npy'))
                    ts = np.load(op.join(message_folder, tg, 'timestamps.npy')) / self.sample_rate
                    channels = np.load(op.join(message_folder, tg, 'channels.npy'))

                    message_data = MessageData(
                        times=ts,
                        channels=channels,
                        text=text,
                    )
                    self._messages.append(message_data)
        elif self.format == 'openephys':
            pass

        self._message_dirty = False


    def _read_channel_groups(self):
        self._channel_id_to_channel_group = {}
        self._channel_group_id_to_channel_group = {}
        self._channel_count = 0
        self._channel_groups = []
        if self.probefile_ch_mapping is not None:
            for channel_group_id, channel_info in self.probefile_ch_mapping.items():
                num_chans = len(channel_info['channels'])
                self._channel_count += num_chans
                channels = []
                for idx, chan in enumerate(channel_info['channels']):
                    channel = Channel(
                        index=idx,
                        channel_id=chan,
                        name="channel_{}_channel_group_{}".format(chan,
                                                                  channel_group_id),
                        gain=channel_info['gain'][idx]
                    )
                    channels.append(channel)

                channel_group = ChannelGroup(
                    channel_group_id=channel_group_id,
                    filename=None,#TODO,
                    channels=channels,
                    fileclass=self,
                    attrs=None #TODO
                )

                self._channel_groups.append(channel_group)
                self._channel_group_id_to_channel_group[channel_group_id] = channel_group

                for chan in channel_info['channels']:
                    self._channel_id_to_channel_group[chan] = channel_group

        # TODO channel mapping to file
        self._channel_ids = np.arange(self._channel_count)
        self._channel_groups_dirty = False


    def _read_events(self):
        if self.format == 'binary':
            events_folder = [op.join(self.absolute_foldername, f)
                                    for f in os.listdir(self.absolute_foldername) if 'events' in f][0]
            processor_folders = [op.join(events_folder, f) for f in os.listdir(events_folder)
                                if 'Tracking_Port' not in f and 'Message_Center' not in f]
            for processor_folder in processor_folders:
                TTL_groups = [f for f in os.listdir(processor_folder) if 'TTL' in f]
                for bg in TTL_groups:
                    full_words = np.load(op.join(processor_folder, bg, 'full_words.npy'))
                    ts = np.load(op.join(processor_folder, bg, 'timestamps.npy'))
                    channels = np.load(op.join(processor_folder, bg, 'channels.npy')).astype(int)
                    channel_states = np.load(op.join(processor_folder, bg, 'channel_states.npy'))
                    channel_states = channel_states/np.max(channel_states).astype(int)
                    metadata_file = op.join(processor_folder, bg, 'metadata.npy')
                    if os.path.exists(metadata_file):
                        metadata = np.load(metadata_file)
                    else:
                        metadata = None

                    ts = ts / self.sample_rate
                    ts -= self.start_time

                    processor_folder_split = op.split(processor_folder)[-1].split("-")

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

                binary_groups = [f for f in os.listdir(processor_folder) if 'binary' in f]
                for bg in binary_groups:
                    full_words = np.load(op.join(processor_folder, bg, 'full_words.npy'))
                    ts = np.load(op.join(processor_folder, bg, 'timestamps.npy'))
                    channels = np.load(op.join(processor_folder, bg, 'channels.npy')).astype(int)
                    channel_states = np.load(op.join(processor_folder, bg, 'channel_states.npy'))
                    channel_states = channel_states / np.max(channel_states).astype(int)
                    metadata_file = op.join(processor_folder, bg, 'metadata.npy')
                    if os.path.exists(metadata_file):
                        metadata = np.load(metadata_file)
                    else:
                        metadata = None

                    ts = ts / self.software_sample_rate
                    ts -= self.start_time

                    processor_folder_split = op.split(processor_folder)[-1].split("-")

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
                channel_states[channel_states==0] = -1
                for proc, id in self.sig_chain.items():
                    if int(id) == int(node):
                        processor = proc
                node_id = int(float(node))
                full_words = None
                metadata = None
                ts -= self.start_time

                event_data = EventData(
                    times=ts,
                    channels=channels,
                    channel_states=channel_states,
                    full_words=full_words,
                    processor=processor,
                    node_id=node_id,
                    metadata=metadata
                )

                self._event_signals.append(event_data)

        self._events_dirty = False


    def _read_tracking(self):
        if 'Sources/Tracking Port' in self.sig_chain.keys():
            if self.format == 'binary':
                # Check and decode files
                events_folder = [op.join(self.absolute_foldername, f)
                                 for f in os.listdir(self.absolute_foldername) if 'events' in f][0]
                tracking_folder = [op.join(events_folder, f) for f in os.listdir(events_folder)
                                   if 'Tracking_Port' in f][0]
                binary_groups = [f for f in os.listdir(tracking_folder)]
                for bg in binary_groups:
                    data_array = np.load(op.join(tracking_folder, bg, 'data_array.npy'))
                    ts = np.load(op.join(tracking_folder, bg, 'timestamps.npy'))
                    channels = np.load(op.join(tracking_folder, bg, 'channels.npy'))
                    metadata = np.load(op.join(tracking_folder, bg, 'metadata.npy'))
                    data_array = np.array([struct.unpack('4f', d) for d in data_array])

                    ts = ts / self.software_sample_rate
                    ts -= self.start_time

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
        if self.experiment.acquisition_system is not None:
            if self.format == 'binary':
                # Check and decode files
                continuous_folder = [op.join(self.absolute_foldername, f)
                                     for f in os.listdir(self.absolute_foldername) if 'continuous' in f][0]
                processor_folder = [op.join(continuous_folder, f) for f in os.listdir(continuous_folder)][0]

                filenames = [f for f in os.listdir(processor_folder)]
                if any('.dat' in f for f in filenames):
                    datfile = [f for f in filenames if '.dat' in f and 'continuous' in f][0]
                    print('.dat: ', datfile)
                    with open(op.join(processor_folder, datfile), "rb") as fh:
                        anas, nsamples = read_analog_binary_signals(fh, self.nchan)
                    ts = np.load(op.join(processor_folder, 'timestamps.npy')) / self.sample_rate
                    if len(ts) != nsamples:
                        print('Warning: timestamps and nsamples are different!')
                        ts = np.arange(nsamples) / self.sample_rate
                    else:
                        ts -= self.start_time
                else:
                    raise ValueError("'continuous.dat' should be in the folder")
            elif self.format == 'openephys':
                # Find continuous CH data
                if self.experiment.id == 1:
                    contFiles = [f for f in os.listdir(self.absolute_foldername) if 'continuous' in f and 'CH' in f
                                 and len(f.split('_'))==2]
                else:
                    contFiles = [f for f in os.listdir(self.absolute_foldername) if 'continuous' in f and 'CH' in f
                                 and '_' + str(self.experiment.id) in f]

                # order channels
                idxs = [int(x[x.find('CH') + 2: x.find('.')]) for x in contFiles]
                contFiles = list(np.array(contFiles)[np.argsort(idxs)])

                if len(contFiles) != 0:
                    print('Reading all channels')
                    anas = np.array([])
                    for i_f, f in enumerate(contFiles):
                        print(f)
                        fullpath = op.join(self.absolute_foldername, f)
                        sig = loadContinuous(fullpath)
                        block_len = int(sig['header']['blockLength'])
                        sample_rate = float(sig['header']['sampleRate'])
                        if anas.shape[0] < 1:
                            anas = sig['data'][None, :]
                        else:
                            if sig['data'].size == anas[-1].size:
                                anas = np.append(anas, sig['data'][None, :], axis=0)
                            else:
                                raise Exception('Channels must have the same number of samples')

                        if i_f == len(contFiles) - 1:
                            # Recordings number
                            rec_num = sig['recordingNumber']
                            timestamps = sig['timestamps']
                            idx_rec = np.where(rec_num == self.id)[0]
                            if len(idx_rec) > 0:
                                idx_start = idx_rec[0]
                                idx_end = idx_rec[-1]
                                t_start = timestamps[idx_start]
                                t_end = timestamps[idx_end] + block_len
                                anas_start = idx_start*block_len
                                anas_end = (idx_end + 1)*block_len
                                ts = np.arange(t_start, t_end) / sample_rate
                                anas = anas[:, anas_start:anas_end]
                    self._processor_sample_rate = sample_rate * pq.Hz
                    nsamples = anas.shape[1]
            # Keep only selected channels
            if self._keep_channels is not None:
                assert anas.shape[1] == nsamples, 'Assumed wrong shape'
                anas_keep = anas[self._keep_channels, :]
            else:
                anas_keep = anas
            self._analog_signals = [AnalogSignal(
                channel_id=range(anas_keep.shape[0]),
                signal=anas_keep,
                times=ts,
            )]
        else:
            self._analog_signals = [AnalogSignal(
                channel_id=np.array([]),
                signal=np.array([]),
                times=np.array([])
            )]

        self._analog_signals_dirty = False


    def _read_spiketrains(self):
        if self.format == 'binary':
            # Check and decode files
            spikes_folder = [op.join(self.absolute_foldername, f)
                             for f in os.listdir(self.absolute_foldername) if 'spikes' in f][0]
            processor_folders = [op.join(spikes_folder, f) for f in os.listdir(spikes_folder)]

            for processor_folder in processor_folders:
                spike_groups = [f for f in os.listdir(processor_folder)]
                for bg in spike_groups:
                    spike_clusters = np.load(op.join(processor_folder, bg, 'spike_clusters.npy'))
                    spike_electrode_indices = np.load(op.join(processor_folder, bg, 'spike_electrode_indices.npy'))
                    spike_times = np.load(op.join(processor_folder, bg, 'spike_times.npy'))
                    spike_waveforms = np.load(op.join(processor_folder, bg, 'spike_waveforms.npy'))

                    metadata_file = op.join(processor_folder, bg, 'metadata.npy')
                    if os.path.exists(metadata_file):
                        metadata = np.load(metadata_file)
                    else:
                        metadata = None

                    spike_times = spike_times / self.sample_rate
                    spike_times -= self.start_time
                    processor_folder_split = op.split(processor_folder)[-1].split("-")

                    clusters = np.unique(spike_clusters)
                    print('Clusters: ', len(clusters))
                    for clust in clusters:
                        idx = np.where(spike_clusters==clust)[0]
                        spiketrain = SpikeTrain(times=spike_times[idx],
                                                waveforms=spike_waveforms[idx],
                                                electrode_indices=spike_electrode_indices[idx],
                                                cluster=clust,
                                                metadata=metadata)
                        self._spiketrains.append(spiketrain)


        elif self.format == 'openephys':
            filenames = [f for f in os.listdir(self.absolute_foldername)
                         if f.endswith('.spikes')]
            # order channels
            idxs = [int(x.split('.')[1][x.split('.')[1].find('0n')+2:]) for x in filenames]
            filenames = list(np.array(filenames)[np.argsort(idxs)])

            if len(filenames) != 0:
                self._spiketrains = []
                spike_clusters = np.array([])
                spike_times = np.array([])
                spike_electrode_indices = np.array([])
                spike_waveforms = np.array([])
                if len(filenames) == 0:
                    return
                for i_f, fname in enumerate(filenames):
                    print('Loading spikes from ', fname)
                    data = loadSpikes(op.join(self.absolute_foldername, fname))

                    if i_f == 0:
                        spike_clusters = np.max(data['sortedId'], axis=1).astype(int)
                        spike_times = data['timestamps']
                        spike_electrode_indices = np.array([int(fname[fname.find('0n')+2]) + 1]
                                                                       * len(spike_clusters))
                        spike_waveforms = data['spikes'].swapaxes(1, 2)
                    else:
                        spike_clusters = np.hstack((spike_clusters, np.max(data['sortedId'], axis=1).astype(int)))
                        spike_times = np.hstack((spike_times, data['timestamps']))
                        spike_electrode_indices = np.hstack((spike_electrode_indices,
                                                                  np.array([int(fname[fname.find('0n')+2]) + 1]
                                                                           * len(data['sortedId']))))
                        spike_waveforms = np.vstack((spike_waveforms, data['spikes'].swapaxes(1, 2)))

                clusters = np.unique(spike_clusters)
                print('Clusters: ', len(clusters))
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

        if clipping_times is not None:
            if clipping_times is not list:
                if type(clipping_times[0]) is not pq.quantity.Quantity:
                    raise AttributeError('clipping_times must be a quantity list of length 1 or 2')

            clipping_times = [t.rescale(pq.s) for t in clipping_times]

            for anas in self.analog_signals:
                anas.signal = clip_anas(anas, self.times, clipping_times, start_end)
            for digs in self.digital_in_signals:
                digs.times = clip_digs(digs, clipping_times, start_end)
                digs.times = digs.times - clipping_times[0]
            for track in self.tracking:
                track.positions, track.times = clip_tracking(track, clipping_times,start_end)

            self._times = clip_times(self._times, clipping_times, start_end)
            self._times -= self._times[0]
            self._duration = self._times[-1] - self._times[0]
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
