# Copyright 2018 Timo Nolle
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================

import pickle as pickle

import numpy as np

from utils.enums import Normalization, Mode, Base, Heuristic, Strategy


class AnomalyDetector(object):
    """Abstract base anomaly detector class.

    This is a boilerplate anomaly detector that only provides simple serialization and deserialization methods
    using pickle. Other classes can inherit the behavior. They will have to implement both the fit and the predict
    method.
    """
    abbreviation = None
    name = None
    supported_binarization = []
    supported_heuristics = []
    supported_strategies = []
    supported_normalizations = [Normalization.MINMAX]
    supported_modes = [Mode.BINARIZE]
    supported_bases = [Base.SCORES]
    supports_attributes = False

    def __init__(self, model=None):
        """Initialize base anomaly detector.

        :param model: Path to saved model file. Defaults to None.
        :type model: str
        """
        self._model = None

    @property
    def model(self):
        return self._model

    def _save(self, file_name):
        """The function to save a model. Subclasses that do not use pickle must override this method."""
        with open(file_name, 'wb') as f:
            pickle.dump(self._model, f)


    def fit(self, dataset):
        """Train the anomaly detector on a dataset.

        This method must be implemented by the subclasses.

        :param dataset: Must be passed as a Dataset object
        :type dataset: Dataset
        :return: None
        """
        raise NotImplementedError()

    def detect(self, dataset):
        """Detect anomalies on an event log.

        This method must be implemented by the subclasses.

        Detects anomalies on a given dataset. Dataset can be passed as in the fit method.
        Returns an array containing an anomaly score for each attribute in each event in each case.

        :param dataset:
        :type dataset: Dataset
        :return: Array of anomaly scores: Shape is [number of cases, maximum case length, number of attributes]
        :rtype: numpy.ndarray
        """
        raise NotImplementedError()

