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

import numpy as np

from baseline.binet.core import NNAnomalyDetector
from utils.enums import Heuristic, Strategy, Mode, Base


class DAE(NNAnomalyDetector):
    """Implements a denoising autoencoder based anomaly detection algorithm."""

    abbreviation = 'dae'
    name = 'DAE'

    supported_heuristics = [Heuristic.BEST, Heuristic.ELBOW_DOWN, Heuristic.ELBOW_UP,
                            Heuristic.LP_LEFT, Heuristic.LP_MEAN, Heuristic.LP_RIGHT,
                            Heuristic.MEAN, Heuristic.MEDIAN, Heuristic.RATIO, Heuristic.MANUAL]
    supported_strategies = [Strategy.SINGLE, Strategy.ATTRIBUTE, Strategy.POSITION, Strategy.POSITION_ATTRIBUTE]
    supported_modes = [Mode.BINARIZE]
    supported_bases = [Base.LEGACY, Base.SCORES]
    supports_attributes = True

    config = dict(hidden_layers=2,
                  hidden_size_factor=.01,
                  noise=None)

    def __init__(self, model=None):
        """Initialize DAE model.

        Size of hidden layers is based on input size. The size can be controlled via the hidden_size_factor parameter.
        This can be float or a list of floats (where len(hidden_size_factor) == hidden_layers). The input layer size is
        multiplied by the respective factor to get the hidden layer size.

        :param model: Path to saved model file. Defaults to None.
        :param hidden_layers: Number of hidden layers. Defaults to 2.
        :param hidden_size_factor: Size factors for hidden layer base don input layer size.
        :param epochs: Number of epochs to train.
        :param batch_size: Mini batch size.
        """
        super(DAE, self).__init__(model=model)

    @staticmethod
    def model_fn(dataset, **kwargs):
        # Import keras locally
        from keras.layers import Input, Dense, Dropout, GaussianNoise
        from keras.models import Model
        from keras.optimizers import adam_v2

        hidden_layers = kwargs.pop('hidden_layers')
        hidden_size_factor = kwargs.pop('hidden_size_factor')
        noise = kwargs.pop('noise')

        features = dataset.flat_onehot_features_2d

        # Parameters
        input_size = features.shape[1]

        # Input layer
        input = Input(shape=(input_size,), name='input')
        x = input

        # Noise layer
        if noise is not None:
            x = GaussianNoise(noise)(x)

        # Hidden layers
        for i in range(hidden_layers):
            if isinstance(hidden_size_factor, list):
                factor = hidden_size_factor[i]
            else:
                factor = hidden_size_factor
            x = Dense(max(int(input_size * factor),64), activation='relu', name=f'hid{i + 1}')(x)
            x = Dropout(0.5)(x)

        # Output layer
        output = Dense(input_size, activation='sigmoid', name='output')(x)

        # Build model
        model = Model(inputs=input, outputs=output)

        # Compile model
        model.compile(
            optimizer=adam_v2.Adam(lr=0.0001, beta_2=0.99),
            loss='mean_squared_error',
        )

        return model, features, features  # Features are also targets

    def detect(self, dataset):
        """
        Calculate the anomaly score for each event attribute in each trace.
        Anomaly score here is the mean squared error.

        :param traces: traces to predict
        :return:
            scores: anomaly scores for each attribute;
                            shape is (#traces, max_trace_length - 1, #attributes)

        """
        # Get features
        _, features, _ = self.model_fn(dataset, **self.config)

        # Parameters
        input_size = int(self.model.input.shape[1])
        features_size = int(features.shape[1])
        if input_size > features_size:
            features = np.pad(features, [(0, 0), (0, input_size - features_size), (0, 0)], mode='constant')
        elif input_size < features_size:
            features = features[:, :input_size]

        predictions=[]
        batch_size = 128
        i=0
        while features.shape[0]>=batch_size*i:
            predictions.append( self.model.predict(features[batch_size*i:batch_size*(i+1)], verbose=True))
            i += 1

        predictions= np.concatenate(predictions, 0)

        # Calculate error
        errors = np.power(dataset.flat_onehot_features_2d - predictions, 2)
        errors = errors * np.expand_dims(~dataset.mask, 2).repeat(dataset.attribute_dims.sum(), 2).reshape(
            dataset.mask.shape[0], -1)

        trace_level_abnormal_scores = errors.sum(1) / (dataset.case_lens * dataset.attribute_dims.sum())

        # Split the errors according to the events
        split_event = np.cumsum(np.tile(dataset.attribute_dims.sum(), [dataset.max_len]), dtype=int)[:-1]
        errors_event = np.split(errors, split_event, axis=1)
        errors_event = np.array([np.mean(a, axis=1) if len(a) > 0 else 0.0 for a in errors_event])
        event_level_abnormal_scores = errors_event.T

        # Split the errors according to the attribute dims
        split = np.cumsum(np.tile(dataset.attribute_dims, [dataset.max_len]), dtype=int)[:-1]
        errors_attr = np.split(errors, split, axis=1)
        errors_attr = np.array([np.mean(a, axis=1) if len(a) > 0 else 0.0 for a in errors_attr])

        # Init anomaly scores array
        attr_level_abnormal_scores = np.zeros(dataset.binary_targets.shape)

        for i in range(len(dataset.attribute_dims)):
            error = errors_attr[i::len(dataset.attribute_dims)]
            attr_level_abnormal_scores[:, :, i] = error.T

        return trace_level_abnormal_scores, event_level_abnormal_scores, attr_level_abnormal_scores
