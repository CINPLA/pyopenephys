import numpy as np
import pyopenephys

import tempfile
import unittest
from pathlib import Path
import quantities as pq
from datalad.api import install, Dataset
from parameterized import parameterized


class TestPyopenephysConversions(unittest.TestCase):
    def setUp(self):
        pt = Path.cwd()/'ephy_testing_data'
        if pt.exists():
            self.dataset = Dataset(pt)
        else:
            self.dataset = install('https://gin.g-node.org/NeuralEnsemble/ephy_testing_data')
        self.savedir = Path(tempfile.mkdtemp())

    def get_data(self, rt_write_fname, rt_read_fname, save_fname, dataset_path):
        if rt_read_fname is None:
            rt_read_fname = rt_write_fname
        save_path = self.savedir/save_fname
        rt_write_path = self.savedir/rt_write_fname
        rt_read_path = self.savedir/rt_read_fname
        resp = self.dataset.get(dataset_path)

        return rt_write_path, rt_read_path, save_path

    @parameterized.expand([
        (
        True,
        "openephys/OpenEphys_SampleData_1",
        Path.cwd() / "ephy_testing_data" / "openephys" / "OpenEphys_SampleData_1"
        ),
        (
        True,
        "openephys/OpenEphys_SampleData_2_(multiple_starts)",
        Path.cwd() / "ephy_testing_data" / "openephys" / "OpenEphys_SampleData_2_(multiple_starts)"
        ),
        (
        True,
        "openephys/OpenEphys_SampleData_3",
        Path.cwd() / "ephy_testing_data" / "openephys" / "OpenEphys_SampleData_1"
        ),
        (
        True,
        "openephysbinary/v0.4.4.1_with_video_tracking",
        Path.cwd() / "ephy_testing_data" / "openephysbinary" / "v0.4.4.1_with_video_tracking"
        ),
        (
        True,
        "openephysbinary/v0.5.3_two_neuropixels_stream/Record_Node_107",
        Path.cwd() / "ephy_testing_data" / "openephysbinary" / "v0.5.3_two_neuropixels_stream"/ "Record_Node_107"
        ),
    ])
    def test_open_file(self, download, dataset_path, foldername):
        if download:
            print(f"Testing GIN {dataset_path}")
            self.dataset.get(dataset_path)
        else:
            print(f"Testing local {foldername}")

        if foldername.is_dir():
            file = pyopenephys.File(foldername)
            print("Instantiated File object")
            experiments = file.experiments
            print(f"\nN experiments: {len(experiments)}")
            for e, exp in enumerate(experiments):
                print(f"\nExperiment {e}")
                recordings = exp.recordings
                print(f"N recordings: {len(recordings)}")

                for r, rec in enumerate(recordings):
                    print(f"\nRecording {r} - duration {rec.duration} - acquisition {rec.experiment.acquisition_system}")
                    analog = rec.analog_signals
                    gains = [an.gains for an in analog]
                    signal_shapes = [an.signal.shape for an in analog]
                    for g, shape in zip(gains, signal_shapes):
                        assert len(g) == shape[0]
                    print(f"N analog signals: {len(analog)} - shapes: {signal_shapes}")
                    events = rec.events
                    print(f"N event signals: {len(events)}")
                    spiketrains = rec.spiketrains
                    print(f"N spiketrains: {len(spiketrains)}")
                    tracking = rec.tracking
                    print(f"N tracking: {len(tracking)}")

                    # test clipping
                    print(f"Test clipping")
                    duration = rec.duration
                    clip_times = 0.2
                    rec.clip_recording(clip_times)
                    clip_times = [0.3, 0.8]
                    rec.clip_recording(clip_times)
        else:
            print(f"{foldername} not found!")


if __name__ == '__main__':
    unittest.main()
