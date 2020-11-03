# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Tests to convert from pulse schedules to signals.
"""

import numpy as np

from qiskit.pulse import (Schedule, DriveChannel, Play, Drag, ShiftFrequency,
                          GaussianSquare, ShiftPhase, Gaussian)
from test.terra.common import QiskitAerTestCase

from qiskit.providers.aer.pulse_new.converters import InstructionToSignals
from qiskit.providers.aer.pulse_new import PiecewiseConstant


class TestPulseToSignals(QiskitAerTestCase):
    """Tests the conversion between pulse schedules and signals."""

    def test_pulse_to_signals(self):
        """Generic test."""

        sched = Schedule(name='Schedule')
        sched += Play(Drag(duration=20, amp=0.5, sigma=4, beta=0.5), DriveChannel(0))
        sched += ShiftPhase(1.0, DriveChannel(0))
        sched += Play(Drag(duration=20, amp=0.5, sigma=4, beta=0.5), DriveChannel(0))
        sched += ShiftFrequency(0.5, DriveChannel(0))
        sched += Play(GaussianSquare(duration=200, amp=0.3, sigma=4, width=150), DriveChannel(0))

        test_gaussian = GaussianSquare(duration=200, amp=0.3, sigma=4, width=150)
        sched = sched.insert(0, Play(test_gaussian, DriveChannel(1)))

        converter = InstructionToSignals(dt=1, carriers=None)

        signals = converter.get_signals(sched)

        self.assertEquals(len(signals), 2)
        self.assertTrue(isinstance(signals[0], PiecewiseConstant))
        self.assertTrue(isinstance(signals[0], PiecewiseConstant))

        samples = test_gaussian.get_waveform().samples
        self.assertTrue(np.allclose(signals[1].samples[0:len(samples)], samples))

    def test_shift_phase_to_signals(self):
        """Test that a shift phase gives negative envelope."""

        gaussian = Gaussian(duration=20, amp=0.5, sigma=4)

        sched = Schedule(name='Schedule')
        sched += ShiftPhase(np.pi, DriveChannel(0))
        sched += Play(gaussian, DriveChannel(0))

        converter = InstructionToSignals(dt=1, carriers=None)
        signals = converter.get_signals(sched)

        self.assertTrue(signals[0].samples[10] < 0)
        self.assertTrue(gaussian.get_waveform().samples[10] > 0)

    def test_carriers_and_dt(self):
        """Test that the carriers go into the signals."""

        sched = Schedule(name='Schedule')
        sched += Play(Gaussian(duration=20, amp=0.5, sigma=4), DriveChannel(0))

        converter = InstructionToSignals(dt=0.222, carriers=[5.5e9])
        signals = converter.get_signals(sched)

        self.assertEquals(signals[0].carrier_freq, 5.5e9)
        self.assertEquals(signals[0]._dt, 0.222)
