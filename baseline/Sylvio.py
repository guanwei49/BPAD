'''
Anomaly Detection on Event Logs with a Scarcity of Labels
'''
import numpy as np
from gensim.models import Word2Vec
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

class W2VLOF():
    def __init__(self):
        self.name= 'W2V-LOF'

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

        self.model = LocalOutlierFactor(n_jobs=8)

        self.model.fit(vectors)

        scores = - self.model.negative_outlier_factor_  # 越接近1，越好， 越接近正无穷，越差

        trace_level_abnormal_scores =  (scores-scores.min())/(scores.max()-scores.min())

        attr_Shape = (dataset.num_cases, dataset.max_len, dataset.num_attributes)
        attr_level_abnormal_scores = np.zeros(attr_Shape)
        event_level_abnormal_scores = attr_level_abnormal_scores.max((2))

        return trace_level_abnormal_scores, event_level_abnormal_scores, attr_level_abnormal_scores
