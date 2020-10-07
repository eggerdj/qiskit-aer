# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from abc import ABC, abstractmethod
from typing import Callable, Union, List, Optional
import numpy as np
from copy import deepcopy

from .signals import VectorSignal, Signal
from .frame import BaseFrame, Frame
from qiskit.quantum_info.operators import Operator

class BaseOperatorModel(ABC):
    """Abstract interface for an operator model.
    """

    @property
    @abstractmethod
    def frame(self) -> BaseFrame:
        """Get the frame."""
        pass

    @frame.setter
    @abstractmethod
    def frame(self, frame):
        """Set the frame."""
        pass

    @property
    @abstractmethod
    def cutoff_freq(self):
        """Get cutoff frequency."""
        pass

    @cutoff_freq.setter
    @abstractmethod
    def cutoff_freq(self, cutoff_freq):
        """Set cutoff frequency."""
        pass

    @abstractmethod
    def evaluate(self, t: float) -> np.array:
        """Evaluate the model at a given time."""
        pass

    def lmult(self,
              time: float,
              y: np.array,
              in_frame_basis: bool = False) -> np.array:
        """
        Return the product evaluate(t) @ y. Default implementation is to
        call evaluate then multiply.

        Args:
            time: Time at which to create the generator.
            y: operator or vector to apply the model to.
            in_frame_basis: whether to evaluate in the frame basis

        Returns:
            np.array: the product
        """
        return np.dot(self.evaluate(time, in_frame_basis), y)

    def rmult(self,
              time: float,
              y: np.array,
              in_frame_basis: bool = False) -> np.array:
        """
        Return the product y @ evaluate(t). Default implementation is to call
        evaluate then multiply.

        Args:
            time: Time at which to create the generator.
            y: operator or vector to apply the model to.
            in_frame_basis: whether to evaluate in the frame basis

        Returns:
            np.array: the product
        """
        return np.dot(y, self.evaluate(time, in_frame_basis))

    @property
    @abstractmethod
    def drift(self) -> np.array:
        """Evaluate the constant part of the model."""
        pass

    @abstractmethod
    def copy(self):
        """Return a copy of self."""
        pass


class OperatorModel(BaseOperatorModel):
    """OperatorModel representing a sum of :class:`Operator` objects with
    time dependent coefficients.

    Specifically, this object represents a time dependent matrix of
    the form:

    .. math::

        G(t) = \sum_{i=0}^{k-1} s_i(t) G_i,

    for :math:`G_i` matrices (represented by :class:`Operator` objects),
    and the :math:`s_i(t)` given by signals represented by a
    :class:`VectorSignal` object, or a list of :class:`Signal` objects.

    This object contains functionality to evaluate :math:`G(t)` for a given
    :math:`t`, or to compute products :math:`G(t)A` and :math:`AG(t)` for
    :math:`A` an :class:`Operator` or array of suitable dimension.

    Additionally, this class has functionality for representing :math:`G(t)`
    in a rotating frame, and setting frequency cutoffs
    (e.g. for the rotating wave approximation).
    Specifically, given an anti-Hermitian frame operator :math:`F` (i.e.
    :math:`F^\dagger = -F`), entering the frame of :math:`F` results in
    the object representing the rotating operator :math:`e^{-Ft}G(t)e^{Ft} - F`.

    Further, if a frequency cutoff is set, when evaluating the
    `OperatorModel`, any terms with a frequency above the cutoff
    (which combines both signal frequency information and frame frequency
    information) will be set to :math:`0`.

    The signals in the model can be specified either directly (by giving a
    list of Signal objects or a VectorSignal), or by specifying a
    signal_mapping, defined as any function with return type
    `Union[List[Signal], VectorSignal]`. In this mode, assignments to the
    signal attribute will be treated as inputs to the signal_mapping. E.g.

    .. code-block:: python

        signal_map = lambda a: [Signal(lambda t: a * t, 1.)]
        model = OperatorModel(operators=[op], signal_mapping=signal_map)

        # setting signals now will pass the value into the signal_map function
        model.signals = 2.

        # the stored signals (retrivable with model.signals) is now
        # the output of signal_map(2.), converted to a VectorSignal

    See the signals property setter for a more detailed description.
    """

    def __init__(self,
                 operators: List[Operator],
                 signals: Optional[Union[VectorSignal, List[Signal]]] = None,
                 signal_mapping: Optional[Callable] = None,
                 frame: Optional[Union[Operator, np.array, Frame]] = None,
                 cutoff_freq: Optional[float] = None):
        """Initialize.

        Args:
            operators: list of Operator objects.
            signals: Specifiable as either a VectorSignal, a list of
                     Signal objects, or as the inputs to signal_mapping.
                     OperatorModel can be instantiated without specifying
                     signals, but it can not perform any actions without them.
            signal_mapping: a function returning either a
                            VectorSignal or a list of Signal objects.

            frame: Rotating frame operator. If specified with a 1d
                            array, it is interpreted as the diagonal of a
                            diagonal matrix.
            cutoff_freq: Frequency cutoff when evaluating the model.
        """

        self._operators = operators

        self._cutoff_freq = cutoff_freq

        # initialize signal-related attributes
        self._signal_params = None
        self._signals = None
        self._carrier_freqs = None
        self.signal_mapping = signal_mapping
        self.signals = signals

        # initialize flag that frame is None to True, then set frame
        self._frame_is_None = True
        self.frame = frame

        # initialize internal operator representation in the frame basis
        self.__ops_in_fb_w_cutoff = None
        self.__ops_in_fb_w_conj_cutoff = None

    @property
    def signals(self) -> VectorSignal:
        """Return the signals in the model."""
        return self._signals

    @signals.setter
    def signals(self, signals: Union[VectorSignal, List[Signal]]):
        """Set the signals.

        Behavior:
            - If no signal_mapping is specified, the argument signals is
              assumed to be either a list of Signal objects, or a VectorSignal,
              and is saved in self._signals.
            - If a signal_mapping is specified, signals is assumed to be a valid
              input of signal_mapping. The argument signals is set to
              self._signal_params, and the output of signal_mapping is saved in
              self._signals.
        """
        if signals is None:
            self._signal_params = None
            self._signals = None
            self._carrier_freqs = None
        else:

            # if a signal_mapping is specified, take signals as the input
            if self.signal_mapping is not None:
                self._signal_params = signals
                signals = self.signal_mapping(signals)

            # if signals is a list, instantiate a VectorSignal
            if isinstance(signals, list):
                signals = VectorSignal.from_signal_list(signals)

            # if it isn't a VectorSignal by now, raise an error
            if not isinstance(signals, VectorSignal):
                raise Exception('signals specified in unaccepted format.')

            # verify signal length is same as operators
            if len(signals.carrier_freqs) != len(self._operators):
                raise Exception("""signals needs to have the same length as
                                    operators.""")

            # update internal carrier freqs and signals
            self.carrier_freqs = signals.carrier_freqs
            self._signals = signals

    @property
    def carrier_freqs(self) -> np.array:
        """Return the internally stored carrier frequencies."""
        return self._carrier_freqs

    @carrier_freqs.setter
    def carrier_freqs(self, carrier_freqs) -> np.array:
        """Set the internally stored carrier frequencies."""
        if any(carrier_freqs != self._carrier_freqs):
            self._carrier_freqs = carrier_freqs
            self._reset_internal_ops()

    @property
    def frame(self) -> Frame:
        """Return the frame."""
        return self._frame

    @frame.setter
    def frame(self, frame: Union[Operator, np.array, Frame]):
        """Set the frame; must be anti-Hermitian. See class
        docstring for the effects of setting a frame.

        Accepts frame_operator as:
            - An Operator object
            - A 2d array
            - A 1d array - in which case it is assumed the frame operator is a
              diagonal matrix, and the array gives the diagonal elements.
        """

        if frame is None:
            self._frame_is_None = True
            self._frame = Frame(np.zeros(self._operators[0].dim[0]))
        else:
            if isinstance(frame, Frame):
                self._frame = frame
            else:
                self._frame = Frame(frame)

            self._frame_is_None = False

        self._reset_internal_ops()

    @property
    def cutoff_freq(self) -> float:
        """Return the cutoff frequency."""
        return self._cutoff_freq

    @cutoff_freq.setter
    def cutoff_freq(self, cutoff_freq: float):
        """Set the cutoff frequency."""
        if cutoff_freq != self._cutoff_freq:
            self._cutoff_freq = cutoff_freq
            self._reset_internal_ops()

    def evaluate(self, time: float, in_frame_basis: bool = False) -> np.array:
        """
        Evaluate the model in array format.

        Args:
            time: Time to evaluate the model
            in_frame_basis: Whether to evaluate in the basis in which the frame
                            operator is diagonal

        Returns:
            np.array: the evaluated model
        """

        if self.signals is None:
            raise Exception("""OperatorModel cannot be
                               evaluated without signals.""")

        sig_vals = self.signals.value(time)

        op_combo = self._evaluate_linear_combo(sig_vals)
        return self.frame.generator_into_frame(time,
                                               op_combo,
                                               operator_in_frame_basis=True,
                                               return_in_frame_basis=in_frame_basis)

    @property
    def drift(self) -> np.array:
        """Return the part of the model with only Constant coefficients as a
        numpy array.
        """

        # for now if the frame operator is not None raise an error
        if not self._frame_is_None:
            raise Exception("""The drift is currently ill-defined if
                               frame_operator is not None.""")

        drift_sig_vals = self.signals.drift_array

        return self._evaluate_linear_combo(drift_sig_vals)

    def copy(self):
        return deepcopy(self)

    def _evaluate_linear_combo(self, coeffs):
        return 0.5 * (np.tensordot(coeffs, self._ops_in_fb_w_cutoff, axes=1)
                      + np.tensordot(coeffs.conj(),
                                     self._ops_in_fb_w_conj_cutoff,
                                     axes=1))

    @property
    def _ops_in_fb_w_cutoff(self):
        if self.__ops_in_fb_w_cutoff is None:
            self._construct_ops_in_fb_w_cutoff()

        return self.__ops_in_fb_w_cutoff

    @property
    def _ops_in_fb_w_conj_cutoff(self):
        if self.__ops_in_fb_w_conj_cutoff is None:
            self._construct_ops_in_fb_w_cutoff()

        return self.__ops_in_fb_w_conj_cutoff

    def _reset_internal_ops(self):
        self.__ops_in_fb_w_cutoff = None
        self.__ops_in_fb_w_conj_cutoff = None

    def _construct_ops_in_fb_w_cutoff(self):
        """Helper function for constructing frame helper from relevant
        attributes.
        """
        carrier_freqs = None
        if self.carrier_freqs is None:
            carrier_freqs = np.zeros(len(self._operators))
        else:
            carrier_freqs = self.carrier_freqs

        self.__ops_in_fb_w_cutoff, self.__ops_in_fb_w_conj_cutoff = (
            self.frame.operators_into_frame_basis_with_cutoff(self._operators,
                                                              self.cutoff_freq,
                                                              carrier_freqs))
