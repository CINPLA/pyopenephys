[![Build Status](https://travis-ci.org/CINPLA/pyopenephys.svg?branch=master)](https://travis-ci.org/CINPLA/pyopenephys)
[![Project Status: Active - The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)

# pyopenephys
Python reader for [Open Ephys](www.open-ephys.org).

## Installation

In order to install the pyopenephys package, open a terminal and run:

`pip install pyopenephys`

If you want to install from sources and get the latest updates, clone the repo and install locally:

```bash
git clone https://github.com/CINPLA/pyopenephys
cd pyopenephys
python setup.py install 
# use 'python setup.py develop' to install fixed bugs 
```

## Basic Usage

Pyopenephys allows the user to load data recorded with [Open Ephys](www.open-ephys.org). Currently, only the **binary** (recommended) and **openephys** (support for this format will be dropped in future releases) are supported. 

The first step is creating a `File` object. It only requires to pass the paht to the recording folder.

```python
import pyopenephys
file = pyopenephys.File("path-to-recording-folder") 
```

The file object contains the different experiments (corresponding to different settings files) and each experiment contains a set of recordings.

```python
# all experiments
experiments = file.experiments
print(len(experiments))

# recordings of first experiment
experiment = experiments[0]
recordings = experiment.recordings
print(len(experiments))

# access first recording
recording = recordings[0]
```

Experiments store some useful information: 
- `experiment.datetime` contains the starting date and time of the experiment creation
- `experiment.sig_chain` is a dictionary containing the processors and nodes in the signal chain
- `experiment.settings` is a dictionary with the parsed setting.xml file
- `experiment.acquisition_system` contains the system used to input continuous data (e.g. 'Rhythm FPGA')

Recordings contain the actual data: 
- `recording.duration` is the duration of the recording (in seconds)
- `recording.sample_rate` is the sampling frequency (in Hz)
- `recording.analog_signals` is list of `AnalogSignal` objects, which in turn have a `signal`, `times` (in s), and `channel_id` fields.
- `recording.events` is list of `EventData` objects, which in turn have a `times` (in s), `channels`, `channel_states`, `full_words`, `processor`, `node_id`, and `metadata`  fields.
- `recording.tracking` is list of `TrackingData` objects , which in turn have a `times` (in s), `x`, `y`, `width`, `height`, `channels`, and `metadata` fields. Tracking data are recorded with the `Tracking` plugin (https://github.com/CINPLA/tracking-plugin) and are save in **binary** format only (not in **openephys** format).
- `recording.spiketrains` is list of `SpikeTrain` objects, which in turn have a `times`, `waveforms`, `electrode_indices`, `clusters` and `metadata` fields. Spiketrains are saved by the `Spike Viewer` sink in the Open Ephys GUI, in combination with either the `Spike Detector` and `Spike Viewer`.


With a few lines of code, the data and relevant information can be easily parsed and accessed:

```python
import pyopenephys
import matplotlib.pylab as plt

file = pyopenephys.File("path-to-recording-folder") 
# experiment 1 (0 in Python)
experiment = file.experiments[0]
# recording 1 
recording = experiment.recordings[0]

print('Duration: ', recording.duration)
print('Sampling Rate: ', recording.sample_rate)

analog_signals = recording.analog_signals
events_data = recording.events
spiketrains = recording.spiketrains
# tracking_data are accessible only using binary format
tracking_data = recording.tracking

# plot analog signal of channel 4
signals = analog_signals[0]
fig_an, ax_an = plt.subplots()
ax_an.plot(signals.times, signals.signal[3])

# plot raster for spike trains
fig_sp, ax_sp = plt.subplots()
for i_s, sp in enumerate(spiketrains):
    ax_sp.plot(sp.times, i_s*np.ones(len(sp.times)), '|')

plt.show()
```
