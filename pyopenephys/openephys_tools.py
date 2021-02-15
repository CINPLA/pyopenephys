'''
This code was adapted from OpenEphys.py (https://github.com/open-ephys/analysis-tools/blob/master/Python3/OpenEphys.py)
'''

import os
import numpy as np

# constants
NUM_HEADER_BYTES = 1024
SAMPLES_PER_RECORD = 1024
BYTES_PER_SAMPLE = 2
RECORD_SIZE = 4 + 8 + SAMPLES_PER_RECORD * BYTES_PER_SAMPLE + 10  # size of each continuous record in bytes
RECORD_MARKER = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 255])

# constants for pre-allocating matrices:
MAX_NUMBER_OF_SPIKES = int(1e6)
MAX_NUMBER_OF_RECORDS = int(1e6)
MAX_NUMBER_OF_EVENTS = int(1e6)

def loadContinuous(filepath, dtype='int16'):  
    assert dtype in ('float', 'int16'), \
        'Invalid data type specified for loadContinous, valid types are float and int16'

    # print("Loading continuous data...")

    ch = {}

    # read in the data
    f = open(filepath, 'rb')

    fileLength = os.fstat(f.fileno()).st_size

    # calculate number of samples
    recordBytes = fileLength - NUM_HEADER_BYTES
    if recordBytes % RECORD_SIZE != 0:
        raise Exception("File size is not consistent with a continuous file: may be corrupt")
    nrec = recordBytes // RECORD_SIZE
    nsamp = nrec * SAMPLES_PER_RECORD
    # pre-allocate samples
    samples = np.zeros(nsamp, dtype)
    timestamps = np.zeros(nrec)
    recordingNumbers = np.zeros(nrec)
    indices = np.arange(0, nsamp + 1, SAMPLES_PER_RECORD, np.dtype(np.int64))

    header = readHeader(f)

    recIndices = np.arange(0, nrec)

    for recordNumber in recIndices:

        timestamps[recordNumber] = np.fromfile(f, np.dtype('<i8'), 1)  # little-endian 64-bit signed integer
        N = np.fromfile(f, np.dtype('<u2'), 1)[0]  # little-endian 16-bit unsigned integer

        # print index

        if N != SAMPLES_PER_RECORD:
            raise Exception('Found corrupted record in block ' + str(recordNumber))

        recordingNumbers[recordNumber] = (np.fromfile(f, np.dtype('>u2'), 1))  # big-endian 16-bit unsigned integer

        if dtype == 'float':  # Convert data to float array and convert bits to voltage.
            data = np.fromfile(f, np.dtype('>i2'), N) * float(header['bitVolts'])  # big-endian 16-bit signed integer, multiplied by bitVolts
        else:  # Keep data in signed 16 bit integer format.
            data = np.fromfile(f, np.dtype('>i2'), N)  # big-endian 16-bit signed integer
        samples[indices[recordNumber]:indices[recordNumber + 1]] = data

        marker = f.read(10)  # dump

    # print recordNumber
    # print index

    ch['header'] = header
    ch['timestamps'] = timestamps
    ch['data'] = samples  # OR use downsample(samples,1), to save space
    ch['recordingNumber'] = recordingNumbers
    f.close()
    return ch


def loadSpikes(filepath):
    '''
    Loads spike waveforms and timestamps from filepath (should be .spikes file)
    '''

    data = {}

    # print('loading spikes...')

    f = open(filepath, 'rb')
    header = readHeader(f)

    if float(header[' version']) < 0.4:
        raise Exception('Loader is only compatible with .spikes files with version 0.4 or higher')

    data['header'] = header
    numChannels = int(header['num_channels'])
    numSamples = 40  # **NOT CURRENTLY WRITTEN TO HEADER**

    spikes = np.zeros((MAX_NUMBER_OF_SPIKES, numSamples, numChannels))
    timestamps = np.zeros(MAX_NUMBER_OF_SPIKES)
    source = np.zeros(MAX_NUMBER_OF_SPIKES)
    gain = np.zeros((MAX_NUMBER_OF_SPIKES, numChannels))
    thresh = np.zeros((MAX_NUMBER_OF_SPIKES, numChannels))
    sortedId = np.zeros((MAX_NUMBER_OF_SPIKES, numChannels))
    electrodeId = np.zeros((MAX_NUMBER_OF_SPIKES, numChannels))
    recNum = np.zeros(MAX_NUMBER_OF_SPIKES)

    currentSpike = 0

    while f.tell() < os.fstat(f.fileno()).st_size:
        eventType = np.fromfile(f, np.dtype('<u1'), 1)  # always equal to 4, discard
        timestamps[currentSpike] = np.fromfile(f, np.dtype('<i8'), 1)
        software_timestamp = np.fromfile(f, np.dtype('<i8'), 1)
        source[currentSpike] = np.fromfile(f, np.dtype('<u2'), 1)
        numChannels = int(np.fromfile(f, np.dtype('<u2'), 1))
        numSamples = int(np.fromfile(f, np.dtype('<u2'), 1))
        sortedId[currentSpike] = np.fromfile(f, np.dtype('<u2'), 1)
        electrodeId[currentSpike] = np.fromfile(f, np.dtype('<u2'), 1)
        channel = np.fromfile(f, np.dtype('<u2'), 1)
        color = np.fromfile(f, np.dtype('<u1'), 3)
        pcProj = np.fromfile(f, np.float32, 2)
        sampleFreq = np.fromfile(f, np.dtype('<u2'), 1)

        waveforms = np.fromfile(f, np.dtype('<u2'), numChannels * numSamples)
        gain[currentSpike, :] = np.fromfile(f, np.float32, numChannels)
        thresh[currentSpike, :] = np.fromfile(f, np.dtype('<u2'), numChannels)
        recNum[currentSpike] = np.fromfile(f, np.dtype('<u2'), 1)

        waveforms_reshaped = np.reshape(waveforms, (numChannels, numSamples))
        waveforms_reshaped = waveforms_reshaped.astype(float)
        waveforms_uv = waveforms_reshaped

        for ch in range(numChannels):
            waveforms_uv[ch, :] -= 32768
            waveforms_uv[ch, :] /= gain[currentSpike, ch] * 1000

        spikes[currentSpike] = waveforms_uv.T

        currentSpike += 1

    data['spikes'] = spikes[:currentSpike, :, :]
    data['timestamps'] = timestamps[:currentSpike]
    data['source'] = source[:currentSpike]
    data['gain'] = gain[:currentSpike, :]
    data['thresh'] = thresh[:currentSpike, :]
    data['recordingNumber'] = recNum[:currentSpike]
    data['sortedId'] = sortedId[:currentSpike]
    data['electrodeId'] = electrodeId[:currentSpike]

    f.close()

    return data


def loadEvents(filepath):
    data = {}

    # print('loading events...')

    f = open(filepath, 'rb')
    header = readHeader(f)

    if float(header[' version']) < 0.4:
        raise Exception('Loader is only compatible with .events files with version 0.4 or higher')

    data['header'] = header

    index = -1

    channel = np.zeros(MAX_NUMBER_OF_EVENTS)
    timestamps = np.zeros(MAX_NUMBER_OF_EVENTS)
    sampleNum = np.zeros(MAX_NUMBER_OF_EVENTS)
    nodeId = np.zeros(MAX_NUMBER_OF_EVENTS)
    eventType = np.zeros(MAX_NUMBER_OF_EVENTS)
    eventId = np.zeros(MAX_NUMBER_OF_EVENTS)
    recordingNumber = np.zeros(MAX_NUMBER_OF_EVENTS)

    while f.tell() < os.fstat(f.fileno()).st_size:
        index += 1

        timestamps[index] = np.fromfile(f, np.dtype('<i8'), 1)
        sampleNum[index] = np.fromfile(f, np.dtype('<i2'), 1)
        eventType[index] = np.fromfile(f, np.dtype('<u1'), 1)
        nodeId[index] = np.fromfile(f, np.dtype('<u1'), 1)
        eventId[index] = np.fromfile(f, np.dtype('<u1'), 1)
        channel[index] = np.fromfile(f, np.dtype('<u1'), 1)
        recordingNumber[index] = np.fromfile(f, np.dtype('<u2'), 1)

    data['channel'] = channel[:index]
    data['timestamps'] = timestamps[:index]
    data['eventType'] = eventType[:index]
    data['nodeId'] = nodeId[:index]
    data['eventId'] = eventId[:index]
    data['recordingNumber'] = recordingNumber[:index]
    data['sampleNum'] = sampleNum[:index]

    f.close()

    return data


def readHeader(f):
    header = {}
    h = f.read(1024).decode().replace('\n', '').replace('header.', '')
    for i, item in enumerate(h.split(';')):
        if '=' in item:
            header[item.split(' = ')[0]] = item.split(' = ')[1]
    return header
