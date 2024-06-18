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
from baseline.Sylvio import W2VLOF
from baseline.VAE.vae import VAE
from baseline.VAEOCSVM.vaeOCSVM import VAEOCSVM
from baseline.WAKE.wake import WAKE
from baseline.dae import DAE
from baseline.bezerra import SamplingAnomalyDetector, NaiveAnomalyDetector
from baseline.binet.binet import BINetv3, BINetv2
from baseline.boehmer import LikelihoodPlusAnomalyDetector
from baseline.leverage import Leverage
from utils.dataset import Dataset

from utils.eval import cal_best_PRF
from utils.fs import EVENTLOG_DIR, ROOT_DIR


def fit_and_eva(dataset_name, ad, label_percent, fit_kwargs=None, ad_kwargs=None):
    if ad_kwargs is None:
        ad_kwargs = {}
    if fit_kwargs is None:
        fit_kwargs = {}

    start_time = time.time()

    print(f'dataset_name: {dataset_name}; label_percent: {label_percent}')
    # Dataset
    dataset = Dataset(dataset_name, label_percent=label_percent)

    # AD
    ad = ad(**ad_kwargs)
    print(ad.name)
    resPath=os.path.join(ROOT_DIR, f'result_{ad.name}_{label_percent}.csv')
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
        dict(ad=WAKE),  ## Multi-perspective, attr-level    --- WAKE: A Weakly Supervised Business Process Anomaly Detection Framework via a Pre-Trained Autoencoder.
    ]

    label_percents=[0.01, 0.1]

    print('number of datasets:' + str(len(dataset_names)))
    for label_percent in label_percents:
        for ad in ads:
            for d in dataset_names:
                p = Process(target=fit_and_eva, kwargs={'dataset_name' : d,  **ad, 'label_percent':label_percent})
                p.start()
                p.join()

    # res = [fit_and_eva(d, **ad) for ad in ads for d in dataset_names]