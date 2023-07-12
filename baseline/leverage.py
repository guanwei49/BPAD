import numpy as np


class Leverage():
    def __init__(self):
        self.name= 'Leverage'
    def fit(self,dataset):
        return self

    def detect(self, dataset):
        activity = dataset.flat_onehot_features[:, :, :dataset.attribute_dims[0]]  #adaptor： only remains control flow
        activity = activity.reshape((activity.shape[0], np.prod(activity.shape[1:])))

        XE = np.matrix(activity)

        HE = XE * np.linalg.pinv(XE.T * XE) * XE.T
        l = np.diagonal(HE)

        N = dataset.case_lens

        Z = (N - np.mean(N)) / np.std(N)

        sigZ = 1 / (1 + np.exp(-Z))

        if np.max(N) > 2.2822 / 0.3422:
            cNmax = -2.2822 + np.power(np.max(N), 0.3422)
        else:
            cNmax = 0

        w = np.power((1 - sigZ), cNmax)

        trace_level_abnormal_scores = w * l

        attr_Shape = (dataset.num_cases, dataset.max_len, dataset.num_attributes)
        attr_level_abnormal_scores = np.zeros(attr_Shape)
        event_level_abnormal_scores = attr_level_abnormal_scores.max((2))

        return trace_level_abnormal_scores, event_level_abnormal_scores, attr_level_abnormal_scores
