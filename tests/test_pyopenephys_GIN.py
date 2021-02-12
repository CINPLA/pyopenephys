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
        "openephys/OpenEphys_SampleData_1",
        dict(foldername=Path.cwd() / "ephy_testing_data" / "openephys" / "OpenEphys_SampleData_1")
        ),
        (
        "openephys/OpenEphys_SampleData_2_(multiple_starts)",
        dict(foldername=Path.cwd() / "ephy_testing_data" / "openephys" / "OpenEphys_SampleData_2_(multiple_starts)")
        ),
        (
        "openephys/OpenEphys_SampleData_3",
        dict(foldername=Path.cwd() / "ephy_testing_data" / "openephys" / "OpenEphys_SampleData_1")
        )
    ])
    def test_open_file(self, dataset_path, file_kwargs):
        print(f"Testing {dataset_path}")
        self.dataset.get(dataset_path)

        file = pyopenephys.File(**file_kwargs)
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
                signal_shapes = [an.signal.shape for an in analog]
                print(f"N analog signals: {len(analog)} - shapes: {signal_shapes}")
                events = rec.events
                print(f"N event signals: {len(events)}")
                spiketrains = rec.spiketrains
                print(f"N spiketrains: {len(spiketrains)}")

                # test clipping
                print(f"Test clipping")
                start_time = rec.start_time.magnitude
                clip_times = start_time + 0.2
                rec.clip_recording(clip_times)
                clip_times = [start_time + 0.3, start_time + 0.5]
                rec.clip_recording(clip_times)


if __name__ == '__main__':
    unittest.main()
