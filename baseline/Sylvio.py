'''
Anomaly Detection on Event Logs with a Scarcity of Labels
'''
import numpy as np
from gensim.models import Word2Vec
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

class Word2vecBPAD():
    def __init__(self):
        self.name= 'Word2vecBPAD'

    def create_models(self, cases, window, min_count):
        '''
        Creates a word2vec model
        '''
        model = Word2Vec(
            window=window,
            min_count=min_count,
            workers=8)
        model.build_vocab(cases)
        model.train(cases, total_examples=len(cases), epochs=10)

        return model

    def average_feature_vector(self, cases, model):
        '''
        Computes average feature vector for each trace
        '''
        vectors = []
        for case in cases:
            case_vector = []
            for token in case:
                try:
                    case_vector.append(model.wv[token])
                except KeyError:
                    pass
            vectors.append(np.array(case_vector).mean(axis=0))

        return vectors

    def fit(self,dataset):
        return self


    def detect(self, dataset):
        cases = []
        for case in dataset.features[0].tolist():
            if 0 in case:
                cases.append(list(map(str, case[:case.index(0)])))
            else:
                cases.append(list(map(str, case)))
        # generate model
        self.word2vecModel = self.create_models(cases, 3, 1)
        # calculating the average feature vector for each sentence (trace)
        vectors = self.average_feature_vector(cases, self.word2vecModel)
        # normalization
        self.scl = StandardScaler()
        vectors = self.scl.fit_transform(vectors)

        contamination_param = len(
            dataset.anomaly_indices) / dataset.num_cases  ### The amount of contamination of the data set

        self.model = LocalOutlierFactor(n_neighbors=10, contamination=contamination_param, n_jobs=8)

        trace_level_abnormal_scores =  self.model.fit_predict(vectors) ## Label is 1 for an inlier and -1 for an outlier according to the LOF score and the contamination parameter.

        trace_level_abnormal_scores[trace_level_abnormal_scores == 1] = 0
        trace_level_abnormal_scores[trace_level_abnormal_scores == -1] = 1

        attr_Shape = (dataset.num_cases, dataset.max_len, dataset.num_attributes)
        attr_level_abnormal_scores = np.zeros(attr_Shape)
        event_level_abnormal_scores = attr_level_abnormal_scores.max((2))

        return trace_level_abnormal_scores, event_level_abnormal_scores, attr_level_abnormal_scores
