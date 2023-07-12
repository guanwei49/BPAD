import os
import traceback
import time
# import mlflow
from multiprocessing import Process

import pandas as pd

from baseline.GAE.gae import GAE
from baseline.GAMA.gama import GAMA
from baseline.GRASPED.grasped import GRASPED
from baseline.LAE.lae import LAE
from baseline.Sylvio import Word2vecBPAD
from baseline.VAE.vae import VAE
from baseline.VAEOCSVM.vaeOCSVM import VAEOCSVM
from baseline.dae import DAE
from baseline.bezerra import SamplingAnomalyDetector, NaiveAnomalyDetector
from baseline.binet.binet import BINetv3, BINetv2
from baseline.boehmer import LikelihoodPlusAnomalyDetector
from baseline.leverage import Leverage
from utils.dataset import Dataset

from utils.eval import cal_best_PRF
from utils.fs import EVENTLOG_DIR, ROOT_DIR


def fit_and_eva(dataset_name, ad, fit_kwargs=None , ad_kwargs=None):
    if ad_kwargs is None:
        ad_kwargs = {}
    if fit_kwargs is None:
        fit_kwargs = {}

    start_time = time.time()

    # Dataset
    dataset = Dataset(dataset_name, beta=0.005)
    print(dataset_name)

    # AD
    ad = ad(**ad_kwargs)
    print(ad.name)
    resPath=os.path.join(ROOT_DIR, f'result_{ad.name}.csv')
    try:
        # Train and save
        ad.fit(dataset, **fit_kwargs)

        trace_level_abnormal_scores,event_level_abnormal_scores,attr_level_abnormal_scores=ad.detect(dataset)

        end_time = time.time()

        run_time=end_time-start_time
        print('run_time')
        print(run_time)


        ##trace level
        trace_p, trace_r, trace_f1, trace_aupr = cal_best_PRF(dataset.case_target, trace_level_abnormal_scores)
        print("trace")
        print(trace_p, trace_r, trace_f1, trace_aupr)

        ##event level
        eventTemp = dataset.binary_targets.sum(2).flatten()
        eventTemp[eventTemp > 1] = 1
        event_p, event_r, event_f1, event_aupr = cal_best_PRF(eventTemp, event_level_abnormal_scores.flatten())
        print("event")
        print(event_p, event_r, event_f1, event_aupr)

        ##attr level
        attr_p, attr_r, attr_f1, attr_aupr = cal_best_PRF(dataset.binary_targets.flatten(),
                                                          attr_level_abnormal_scores.flatten())
        print("attr")
        print(attr_p, attr_r, attr_f1,attr_aupr)

        datanew = pd.DataFrame([{'index':dataset_name,'trace_p': trace_p, "trace_r": trace_r,'trace_f1':trace_f1,'trace_aupr':trace_aupr,
                                 'event_p': event_p, "event_r": event_r, 'event_f1': event_f1, 'event_aupr': event_aupr,
                                 'attr_p': attr_p, "attr_r": attr_r, 'attr_f1': attr_f1, 'attr_aupr': attr_aupr,'time':run_time
                                 }])
        if os.path.exists(resPath):
            data = pd.read_csv(resPath)
            data = data.append(datanew,ignore_index=True)
        else:
            data = datanew
        data.to_csv(resPath ,index=False)
    except Exception as e:
        traceback.print_exc()
        datanew = pd.DataFrame([{'index': dataset_name}])
        if os.path.exists(resPath):
            data = pd.read_csv(resPath)
            data = data.append(datanew, ignore_index=True)
        else:
            data = datanew
        data.to_csv(resPath, index=False)





if __name__ == '__main__':
    dataset_names = os.listdir(EVENTLOG_DIR)
    dataset_names.sort()
    if 'cache' in dataset_names:
        dataset_names.remove('cache')

    dataset_names_syn = [name for name in dataset_names if (
                                                        'gigantic' in name
                                                        or 'huge' in name
                                                        or 'large' in name
                                                        or 'medium' in name
                                                        or 'p2p' in name
                                                        or 'paper' in name
                                                        or 'small' in name
                                                        or 'wide' in name
    )]

    dataset_names_real = list(set(dataset_names)-set(dataset_names_syn))
    dataset_names_real.sort()

    ads = [
        dict(ad=LikelihoodPlusAnomalyDetector),  ## Multi-perspective, attr-level 1   --- Multi-perspective anomaly detection in business process execution events (extended to support the use of external threshold)
        dict(ad=NaiveAnomalyDetector),  # control flow, trace-level 1   ---Algorithms for anomaly detection of traces in logs of process aware information systems
        dict(ad=SamplingAnomalyDetector),  # control flow, trace-level 1   ---Algorithms for anomaly detection of traces in logs of process aware information systems
        dict(ad=DAE, fit_kwargs=dict(epochs=20, batch_size=64)),  ## Multi-perspective, attr-level 1   ---Analyzing business process anomalies using autoencoders
        dict(ad=BINetv3, fit_kwargs=dict(epochs=20, batch_size=64)), ## Multi-perspective, attr-level 1 ---BINet: Multi-perspective business process anomaly classification
        dict(ad=BINetv2, fit_kwargs=dict(epochs=20, batch_size=64)), ## Multi-perspective, attr-level 1 ---BINet: Multivariate business process anomaly detection using deep learning
        dict(ad=GAMA,ad_kwargs=dict(n_epochs=20)), ## Multi-perspective, attr-level    ---GAMA: A Multi-graph-based Anomaly Detection Framework for Business Processes via Graph Neural Networks
        dict(ad=VAE), ## Multi-perspective, attr-level 自己修改后使其能够检测attr-level   1   ---Autoencoders for improving quality of process event logs
        dict(ad=LAE), ## Multi-perspective, attr-level  自己修改后使其能够检测attr-level   1   ---Autoencoders for improving quality of process event logs
        dict(ad=GAE), ## Multi-perspective, trace-level    1   ---Graph Autoencoders for Business Process Anomaly Detection
        dict(ad=GRASPED), ## Multi-perspective, attr-level   1 ---GRASPED: A GRU-AE Network Based Multi-Perspective Business Process Anomaly Detection Model
        dict(ad=Leverage), # control flow, trace-level   1    ---Keeping our rivers clean: Information-theoretic online anomaly detection for streaming business process events
        dict(ad=Word2vecBPAD), # control flow, trace-level  1   ---Anomaly Detection on Event Logs with a Scarcity of Labels
        dict(ad=VAEOCSVM) # control flow, trace-level   ---Variational Autoencoder for Anomaly Detection in Event Data in Online Process Mining
    ]


    print(dataset_names)
    for ad in ads:
        for d in dataset_names:
            p = Process(target=fit_and_eva, kwargs={ 'dataset_name' : d,  **ad })
            p.start()
            p.join()

    # res = [fit_and_eva(d, **ad) for ad in ads for d in dataset_names]