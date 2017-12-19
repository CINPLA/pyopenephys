"""
Python library for reading OpenEphys files.
Depends on: sys
            os
            glob
            datetime
            numpy
            quantities
            xmljson

Authors: Alessio Buccino @CINPLA,
         Svenn-Arne Dragly @CINPLA,
         Milad H. Mobarhan @CINPLA,
         Mikkel E. Lepperod @CINPLA
"""

# TODO: add extensive function descrption and verbose option for prints

from __future__ import division
from __future__ import print_function
from __future__ import with_statement

import quantities as pq
import os
import os.path as op
import numpy as np
from datetime import datetime
import locale
import struct
import platform
import xml.etree.ElementTree as ET
from xmljson import yahoo as yh
from tools import *

# TODO related files
# TODO append .continuous files directly to file and memory map in the end
# TODO ChannelGroup class - needs probe file
# TODO Channel class
# TODO add SYNC and TRACKERSTIM metadata


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
        return "<OpenEphys analog signal:shape: {}, sample_rate: {}>".format(
            self.signal.shape, self.sample_rate
        )


class DigitalSignal:
    def __init__(self, times, channel_id, sample_rate):
        self.times = times
        self.channel_id = channel_id
        self.sample_rate = sample_rate

    def __str__(self):
        return "<OpenEphys digital signal: nchannels: {}>".format(
            self.channel_id
        )


class Sync:
    def __init__(self, times, channel_id, sample_rate):
        self.times = times
        self.channel_id = channel_id
        self.sample_rate = sample_rate

    def __str__(self):
        return "<OpenEphys sync signal: nchannels: {}>".format(
            self.channel_id
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


class SpikeTrain:
    def __init__(self, times, waveforms,
                 spike_count, channel_count, samples_per_spike,
                 sample_rate, t_stop, **attrs):
        assert(waveforms.shape[0] == spike_count), waveforms.shape[0]
        assert(waveforms.shape[1] == channel_count), waveforms.shape[1]
        assert(waveforms.shape[2] == samples_per_spike), waveforms.shape[2]
        assert(len(times) == spike_count)
        assert times[-1] <= t_stop, ('Spike time {}'.format(times[-1]) +
                                     ' exceeds duration {}'.format(t_stop))
        self.times = times
        self.waveforms = waveforms
        self.attrs = attrs
        self.t_stop = t_stop

        self.spike_count = spike_count
        self.channel_count = channel_count
        self.samples_per_spike = samples_per_spike
        self.sample_rate = sample_rate

    @property
    def num_spikes(self):
        """
        Alias for spike_count.
        """
        return self.spike_count

    @property
    def num_chans(self):
        """
        Alias for channel_count.
        """
        return self.channel_count

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
                                               sample_rate=ana.sample_rate))
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
        self._absolute_foldername = foldername
        self._path, relative_foldername = os.path.split(foldername)
        self._experiments_names = [f for f in os.listdir(self._absolute_foldername)
                                   if os.path.isdir(op.join(self._absolute_foldername, f))
                                   and 'experiment' in f]
        self._exp_ids = [int(exp[-1]) for exp in self._experiments_names]
        self._experiments = []

        for (rel_path, id) in zip(self._experiments_names, self._exp_ids):
            self._experiments.append(Experiment(op.join(self._absolute_foldername, rel_path), id, probefile))

    @property
    def session(self):
        return os.path.split(self._absolute_foldername)[-1]

    @property
    def datetime(self):
        return self._start_datetime

    @property
    def experiments(self):
        return self._experiments


class Experiment:
    def __init__(self, path, id, probefile=None):
        self._recordings = []
        self._prb_file = probefile
        self._absolute_foldername = path
        self._absolute_parentfolder = op.dirname(path)
        self.id = id
        self.sig_chain = dict()

        self._read_settings(id)
        self._recording_names = [f for f in os.listdir(self._absolute_foldername)
                                   if os.path.isdir(op.join(self._absolute_foldername, f))
                                   and 'recording' in f]
        self._rec_ids = [int(rec[-1]) for rec in self._recording_names]
        for (rel_path, id) in zip(self._recording_names, self._rec_ids):
            self._recordings.append(Recording(op.join(self._absolute_foldername, rel_path), id,
                                              self.sig_chain, self._prb_file))

    def _read_settings(self, id):
        print('Loading Open-Ephys: reading settings.xml...')
        if id == 1:
            set_fname = [fname for fname in os.listdir(self._absolute_parentfolder)
                         if fname == 'settings.xml']
        else:
            set_fname = [fname for fname in os.listdir(self._absolute_parentfolder)
                         if fname.startswith('settings') and fname.endswith('.xml') and str(id) in fname]
        self.rhythm = False
        self.rhythmID = []
        # rhythmRates = np.array([1., 1.25, 1.5, 2, 2.5, 3, 3.33, 4., 5., 6.25,
        #                         8., 10., 12.5, 15., 20., 25., 30.])
        self.track_port = False
        self.track_portID = []
        self.track_portInfo = dict()

        self.track_stim = False
        self.track_stimInfo = []

        self.fileReader = False

        self.sync = False
        self.syncID = []

        if not len(set_fname) == 1:
            raise IOError('Unique settings file not found')

        self._set_fname = op.join(self._absolute_parentfolder, set_fname[0])
        with open(self._set_fname) as f:
            xmldata = f.read()
            self.settings = yh.data(ET.fromstring(xmldata))['SETTINGS']
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
        FPGA_count = 0
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
                self.sig_chain.update({processor['name']: True})
                if processor['name'] == 'Sources/Rhythm FPGA':
                    if FPGA_count > 0:
                        raise NotImplementedError
                        # TODO can there be multiple FPGAs ?
                    FPGA_count += 1
                    self._channel_info['gain'] = {}
                    self.rhythm = True
                    self.rhythmID = processor['NodeId']
                    # gain for all channels
                    gain = {ch['number']: float(ch['gain']) * pq.uV  # TODO assert is uV
                            for chs in processor['CHANNEL_INFO'].values()
                            for ch in chs}
                    for chan in processor['CHANNEL']:
                        if chan['SELECTIONSTATE']['record'] == '1':
                            self.nchan += 1
                            chnum = chan['number']
                            self._channel_info['gain'][chnum] = gain[chnum]

                # debug
                if processor['name'] == 'Sources/File Reader':
                    if FPGA_count > 0:
                        raise NotImplementedError
                        # TODO can there be multiple FPGAs ?
                    FPGA_count += 1
                    self._channel_info['gain'] = {}
                    self.fileReader = True
                    self.fileReaderID = processor['NodeId']
                    for chan in processor['CHANNEL']:
                        if chan['SELECTIONSTATE']['record'] == '1':
                            self.nchan += 1
                if processor['name'] == 'Sources/Tracking Port':
                    self.track_port = True
                    self.track_portID = processor['NodeId']
                    self.track_portInfo.update(processor['TrackingNode'])
                if processor['name'] == 'Sources/Sync Port':
                    self.sync = True
                    self.syncID = processor['NodeId']
                if processor['name'] == 'Sinks/Tracker Stimulator':
                    self.track_stim = True
                    self.track_stimID = processor['NodeId']
                    # new version
                    self.track_stimInfo = dict()
                    self.track_stimInfo['circles'] = processor['CIRCLES']
                    self.track_stimInfo['channels'] = processor['CHANNELS']
                    self.track_stimInfo['output'] = processor['SELECTEDCHANNEL']
                    self.track_stimInfo['sync'] = {'ttl': processor['EDITOR']['TRACKERSTIMULATOR']['syncTTLchan'],
                                                   'output': processor['EDITOR']['TRACKERSTIMULATOR'][
                                                       'syncStimchan']}

        # Check openephys format
        if self.settings['CONTROLPANEL']['recordEngine'] == 'OPENEPHYS':
            self._format = 'openephys'
        elif self.settings['CONTROLPANEL']['recordEngine'] == 'RAWBINARY':
            self._format = 'binary'
        else:
            self._format = None
        print('Decoding data from ', self._format, ' format')

        if self.rhythm:
            print('RhythmFPGA with ', self.nchan, ' channels. NodeId: ', self.rhythmID)
        if self.track_port:
            print('Tracking Port. NodeId: ', self.track_portID)

        if self.rhythm or self.fileReader:
            recorded_channels = sorted([int(chan) for chan in
                                        self._channel_info['gain'].keys()])
            self._channel_info['channels'] = recorded_channels
            if self._prb_file is not None:
                self._keep_channels = []
                self._probefile_ch_mapping = _read_python(self._prb_file)['channel_groups']
                for group_idx, group in self._probefile_ch_mapping.items():
                    group['gain'] = []
                    # prb file channels are sequential, 'channels' are not as they depend on FPGA channel selection
                    # -> Collapse them into array
                    for chan, oe_chan in zip(group['channels'],
                                             group['oe_channels']):
                        if oe_chan not in recorded_channels:
                            raise ValueError('Channel "' + str(oe_chan) +
                                             '" in channel group "' +
                                             str(group_idx) + '" in probefile' +
                                             self._prb_file +
                                             ' is not marked as recorded ' +
                                             'in settings file' +
                                             self._set_fname)
                        group['gain'].append(
                            self._channel_info['gain'][str(oe_chan)]
                        )
                        self._keep_channels.append(recorded_channels.index(oe_chan))
                print('Number of selected channels: ', len(self._keep_channels))
            else:
                self._keep_channels = None  # HACK
                # TODO sequential channel mapping
                print('sequential channel mapping')
        else:
            self._keep_channels = None

        if self.fileReader:
            recorded_channels = sorted([int(chan) for chan in
                                        self._channel_info['gain'].keys()])
            self._channel_info['channels'] = recorded_channels
            if self._prb_file is not None:
                self._keep_channels = []
                self._probefile_ch_mapping = _read_python(self._prb_file)['channel_groups']
                for group_idx, group in self._probefile_ch_mapping.items():
                    group['gain'] = []
                    # prb file channels are sequential, 'channels' are not as they depend on FPGA channel selection -> Collapse them into array
                    for chan, oe_chan in zip(group['channels'],
                                             group['oe_channels']):
                        if oe_chan not in recorded_channels:
                            raise ValueError('Channel "' + str(oe_chan) +
                                             '" in channel group "' +
                                             str(group_idx) + '" in probefile' +
                                             self._prb_file +
                                             ' is not marked as recorded ' +
                                             'in settings file' +
                                             self._set_fname)
                        group['gain'].append(
                            self._channel_info['gain'][str(oe_chan)]
                        )
                        self._keep_channels.append(recorded_channels.index(oe_chan))
                print('Number of selected channels: ', len(self._keep_channels))
            else:
                self._keep_channels = None  # HACK
                # TODO sequential channel mapping
                print('sequential channel mapping')

        self.sig_chain.update({'format': self._format, 'nchan': self.nchan, 'keep_channels': self._keep_channels})

    @property
    def recordings(self):
        return self._recordings


class Recording:
    def __init__(self, path, id, sig_chain, probefile=None):
        self.absolute_foldername = path
        self.prb_file = probefile
        self.sig_chain = sig_chain
        self._format = sig_chain['format']
        self._keep_channels = sig_chain['keep_channels']
        self.nchan = sig_chain['nchan']
        self.id = id

        self._analog_signals_dirty = True
        self._digital_signals_dirty = True
        self._channel_groups_dirty = True
        self._spiketrains_dirty = True
        self._tracking_dirty = True
        self._events_dirty = True
        self._times = []
        self._duration = []

        self._analog_signals = []
        self._digital_signals = []
        self._tracking_signals = []
        self._messages_signals = []

        self.__dict__.update(self._read_sync_message())


    @property
    def duration(self):
        if self.rhythm:
            self._duration = (self.analog_signals[0].signal.shape[1] /
                              self.analog_signals[0].sample_rate)
        elif self.track_port:
            self._duration = self.tracking[0].times[0][-1] - self.tracking[0].times[0][0]
        else:
            self._duration = 0

        return self._duration

    @property
    def sample_rate(self):
        if self.processor:
            return self._processor_sample_rate
        else:
            return self._software_sample_rate

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
    def analog_signals(self):
        if self._analog_signals_dirty:
            self._read_analog_signals()

        return self._analog_signals

    @property
    def spiketrains(self):
        if self._spiketrains_dirty:
            self._read_spiketrains()

        return self._spiketrains

    @property
    def digital_in_signals(self):
        if self._digital_signals_dirty:
            self._read_digital_signals()

        return self._digital_signals

    @property
    def sync_signals(self):
        if self._digital_signals_dirty:
            self._read_digital_signals()

        return self._sync_signals

    @property
    def events(self):
        if self._events_dirty:
            self._read_digital_signals()

        return self._events

    @property
    def tracking(self):
        if self._tracking_dirty:
            self._read_tracking()

        return self._tracking_signals

    @property
    def times(self):
        if self.rhythmID:
            self._times = self.analog_signals[0].times
        elif self.track_port:
            self._times = self.tracking[0].times[0]
        else:
            self._times = []

        return self._times

    def _read_sync_message(self):
        info = dict()
        stimes = []
        sync_messagefile = [f for f in os.listdir(self.absolute_foldername) if 'sync_messages' in f][0]
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
        events_folder = [op.join(self.absolute_foldername, f)
                         for f in os.listdir(self.absolute_foldername) if 'events' in f][0]
        message_folder = [op.join(events_folder, f) for f in os.listdir(events_folder)
                           if 'Message_Center' in f][0]
        text_groups = [f for f in os.listdir(message_folder)]
        if self._format == 'binary':
            for tg in text_groups:
                text = np.load(op.join(message_folder, tg, 'text.npy'))
                ts = np.load(op.join(message_folder, tg, 'timestamps.npy'))
                channels = np.load(op.join(message_folder, tg, 'channels.npy'))


    def _read_channel_groups(self):
        self._channel_id_to_channel_group = {}
        self._channel_group_id_to_channel_group = {}
        self._channel_count = 0
        self._channel_groups = []
        for channel_group_id, channel_info in self._probefile_ch_mapping.items():
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

    def _read_tracking(self):
        if 'Sources/Tracking Port' in self.sig_chain.keys():
            # Check and decode files
            events_folder = [op.join(self.absolute_foldername, f)
                                 for f in os.listdir(self.absolute_foldername) if 'events' in f][0]
            tracking_folder = [op.join(events_folder, f) for f in os.listdir(events_folder)
                                if 'Tracking_Port' in f][0]
            binary_groups = [f for f in os.listdir(tracking_folder)]
            if self._format == 'binary':
                import struct
                for bg in binary_groups:
                    data_array = np.load(op.join(tracking_folder, bg, 'data_array.npy'))
                    ts = np.load(op.join(tracking_folder, bg, 'timestamps.npy'))
                    channels = np.load(op.join(tracking_folder, bg, 'channels.npy'))
                    metadata = np.load(op.join(tracking_folder, bg, 'metadata.npy'))
                    data_array = np.array([struct.unpack('4f', d) for d in data_array])

                    ts = ts / float(self.sample_rate)
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
                self._tracking_dirty = False

    def _read_analog_signals(self):
        if self.processor:
            # Check and decode files
            continuous_folder = [op.join(self.absolute_foldername, f)
                                 for f in os.listdir(self.absolute_foldername) if 'continuous' in f][0]
            processor_folder = [op.join(continuous_folder, f) for f in os.listdir(continuous_folder)
                                if 'File_Reader' in f or 'RhythmFPGA' in f][0]

            filenames = [f for f in os.listdir(processor_folder)]
            if self._format == 'binary':
                if any('.dat' in f for f in filenames):
                    datfile = [f for f in filenames if '.dat' in f and 'continuous' in f][0]
                    print('.dat: ', datfile)
                    with open(op.join(processor_folder, datfile), "rb") as fh:
                        anas, nsamples = read_analog_binary_signals(fh, self.nchan)
                    ts = np.load(op.join(processor_folder, 'timestamps.npy')) / self.sample_rate
                else:
                    raise ValueError("'continuous.dat' should be in the folder")
            elif self._format == 'openephys':
                # Find continuous CH data
                contFiles = [f for f in os.listdir(self._absolute_foldername) if 'continuous' in f and 'CH' in f]
                contFiles = sorted(contFiles)
                if len(contFiles) != 0:
                    print('Reading all channels')
                    anas = np.array([])
                    for f in contFiles:
                        fullpath = op.join(self._absolute_foldername, f)
                        sig = read_analog_continuous_signal(fullpath)
                        if anas.shape[0] < 1:
                            anas = sig['data'][None, :]
                        else:
                            if sig['data'].size == anas[-1].size:
                                anas = np.append(anas, sig['data'][None, :], axis=0)
                            else:
                                raise Exception('Channels must have the same number of samples')
                    assert anas.shape[0] == len(self._channel_info['channels'])
                    nsamples = anas.shape[1]
                    print('Done!')
            # Keep only selected channels
            if self._keep_channels is not None:
                assert anas.shape[1] == nsamples, 'Assumed wrong shape'
                anas_keep = anas[self._keep_channels, :]
            else:
                anas_keep = anas
            self._analog_signals = [AnalogSignal(
                channel_id=range(anas_keep.shape[0]),
                signal=anas_keep,
                times=ts
            )]
        else:
            self._analog_signals = [AnalogSignal(
                channel_id=np.array([]),
                signal=np.array([]),
                times=np.array([])
            )]

        self._analog_signals_dirty = False

    def _read_spiketrains(self):
        if self.rhythm:
            # TODO check if spiketains are recorded from setings
            filenames = [f for f in os.listdir(self._absolute_foldername)
                         if f.endswith('.spikes')]
            self._spiketrains = []
            if len(filenames) == 0:
                return
            for fname in filenames:
                print('Loading spikes from ', fname.split('.')[0])
                data = loadSpikes(op.join(self._absolute_foldername, fname))
                clusters = data['recordingNumber']
                group_id = int(np.unique(data['source']))
                assert 'TT{}'.format(group_id) in fname
                for cluster in np.unique(clusters):
                    wf = data['spikes'][clusters == cluster]
                    wf = wf.swapaxes(1, 2)
                    sample_rate = int(data['header']['sampleRate'])
                    times = data['timestamps'][clusters == cluster]
                    times = (times - self.start_timestamp) / sample_rate
                    t_stop = self.duration.rescale('s')
                    self._spiketrains.append(
                        SpikeTrain(
                            times=times * pq.s,
                            waveforms=wf * pq.uV,
                            spike_count=len(times),
                            channel_count=int(data['header']['num_channels']),
                            sample_rate=sample_rate * pq.Hz,
                            channel_group_id=group_id,
                            samples_per_spike=40,  # TODO read this from file
                            gain=data['gain'][clusters == cluster],
                            threshold=data['thresh'][clusters == cluster],
                            name='Unit #{}'.format(cluster),
                            cluster_id=int(cluster),
                            t_stop=t_stop
                        )
                    )

        self._spiketrains_dirty = False

    def _read_digital_signals(self):
        filenames = [f for f in os.listdir(self._absolute_foldername)]
        if any('.events' in f and 'all_channels' in f for f in filenames):
            eventsfile = [f for f in filenames if '.events' in f and 'all_channels' in f][0]
            print('.events ', eventsfile)
            with open(op.join(self._absolute_foldername, eventsfile), "rb") as fh: #, encoding='utf-8', errors='ignore') as fh:
                data = {}

                print('loading events...')
                header = readHeader(fh)

                if float(header['version']) < 0.4:
                    raise Exception('Loader is only compatible with .events files with version 0.4 or higher')

                data['header'] = header

                struct_fmt = '=qH4BH'  # int[5], float, byte[255]
                struct_len = struct.calcsize(struct_fmt)
                struct_unpack = struct.Struct(struct_fmt).unpack_from

                nsamples = (os.fstat(fh.fileno()).st_size - fh.tell()) // struct_len
                print('Estimated events samples: ', nsamples)
                nread = 0

                read_data = []
                while True:
                    byte = fh.read(struct_len)
                    if not byte:
                        break
                    s = struct_unpack(byte)
                    read_data.append(s)
                    nread += 1

                print('Read event samples: ', nread)

                timestamps, sampleNum, eventType, nodeId, eventId, channel, recordingNumber = zip(*read_data)

                data['channel'] = np.array(channel)
                data['timestamps'] = np.array(timestamps)
                data['eventType'] = np.array(eventType)
                data['nodeId'] = np.array(nodeId)
                data['eventId'] = np.array(eventId)
                data['recordingNumber'] = np.array(recordingNumber)
                data['sampleNum'] = np.array(sampleNum)

                # TODO: check if data is null (data['event...'] is null?
                # Consider only TTL from FPGA (for now)
                num_channels = 8
                self._digital_signals = None
                if self.rhythm:
                    if len(data['timestamps']) != 0:
                        idxttl_fpga = np.where((data['eventType'] == 3) &
                                               (data['nodeId'] == int(self.rhythmID)))
                        digs = [list() for i in range(num_channels)]
                        if len(idxttl_fpga[0]) != 0:
                            print('TTLevents: ', len(idxttl_fpga[0]))
                            digchan = np.unique(data['channel'][idxttl_fpga])
                            print('Used IO channels ', digchan)
                            for chan in digchan:
                                idx_chan = np.where(data['channel'] == chan)
                                dig = data['timestamps'][idx_chan]
                                # Consider rising edge only
                                dig = dig[::2]
                                dig = dig - self.start_timestamp
                                dig = dig.astype('float') / self.sample_rate
                                digs[chan] = dig.rescale('s')

                        self._digital_signals = [DigitalSignal(
                            channel_id=list(range(num_channels)),
                            times=digs,
                            sample_rate=self.sample_rate
                        )]
                if self._digital_signals is None:
                    self._digital_signals = [DigitalSignal(
                        channel_id=np.array([]),
                        times=np.array([]),
                        sample_rate=[]
                    )]

                if self.sync:
                    if len(data['timestamps']) != 0:
                        idxttl_sync = np.where((data['eventType'] == 3) & (data['nodeId'] == int(self.syncID)))
                        syncchan = []
                        syncs = []
                        if len(idxttl_sync[0]) != 0:
                            print('TTL Sync events: ', len(idxttl_sync[0]))
                            syncchan = np.unique(data['channel'][idxttl_sync])
                            # TODO this should be reduced to a single loop
                            if len(syncchan) == 1:
                                # Single digital input
                                syncs = data['timestamps'][idxttl_sync]
                                # remove start_time (offset) and transform in seconds
                                syncs = syncs - self.start_timestamp
                                syncs = syncs.astype(dtype='float') / self.sample_rate
                                syncs = np.array([syncs]) * pq.s
                            else:
                                for chan in syncchan:
                                    idx_chan = np.where(data['channel'] == chan)
                                    new_sync = data['timestamps'][idx_chan]

                                    new_sync = new_sync - self.start_timestamp
                                    new_sync = new_sync.astype(dtype='float') / self.sample_rate
                                    syncs.append(new_sync)
                                syncs = np.array(syncs) * pq.s

                        self._sync_signals = [Sync(
                            channel_id=syncchan,
                            times=syncs,
                            sample_rate=self.sample_rate
                        )]
                    else:
                        self._sync_signals = [DigitalSignal(
                            channel_id=np.array([]),
                            times=np.array([]),
                            sample_rate=[]
                        )]
                else:
                    self._sync_signals = [Sync(
                        channel_id=np.array([]),
                        times=np.array([]),
                        sample_rate=[]
                    )]

                self._digital_signals_dirty = False
                self._events_dirty = False
                # self._events = data


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
