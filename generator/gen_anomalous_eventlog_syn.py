import os
from pathlib import Path

from tqdm import tqdm

from generator.generation.anomaly import *
from generator.generation.utils import generate_for_process_model


def get_process_model_files(path=None):
    # Base
    ROOT_DIR = Path(__file__).parent

    if path is None:
        path = os.path.join(ROOT_DIR/'process_models') # Randomly generated process models from PLG2
    return [os.path.join(path,f) for f in os.listdir(path)]


anomalies = [
    SkipSequenceAnomaly(max_sequence_size=2),##最多跳过的子序列个数
    ReworkAnomaly(max_distance=5, max_sequence_size=3),##重做的和原来的最大距离，重做的最多的子序列个数
    EarlyAnomaly(max_distance=5, max_sequence_size=2),
    LateAnomaly(max_distance=5, max_sequence_size=2),
    InsertAnomaly(max_inserts=2),
    AttributeAnomaly(max_events=3, max_attributes=2)
]


process_models = [m for m in get_process_model_files()]



for process_model in tqdm(process_models, desc='Generate'):
    generate_for_process_model(process_model, size=5000, anomalies=anomalies, anomaly_p=0.05, num_attr=[1, 2, 3, 4], seed=1337)
    generate_for_process_model(process_model, size=5000, anomalies=anomalies, anomaly_p=0.1, num_attr=[1, 2, 3, 4], seed=1337)
    generate_for_process_model(process_model, size=5000, anomalies=anomalies,anomaly_p=0.15, num_attr=[1, 2, 3, 4], seed=1337)
    generate_for_process_model(process_model, size=5000, anomalies=anomalies,anomaly_p=0.2, num_attr=[1, 2, 3, 4], seed=1337)
    generate_for_process_model(process_model, size=5000, anomalies=anomalies,anomaly_p=0.25, num_attr=[1, 2, 3, 4], seed=1337)
    generate_for_process_model(process_model, size=5000, anomalies=anomalies, anomaly_p=0.3, num_attr=[1, 2, 3, 4],
                                seed=1337)
    generate_for_process_model(process_model, size=5000, anomalies=anomalies, anomaly_p=0.35, num_attr=[1, 2, 3, 4],
                               seed=1337)
    generate_for_process_model(process_model, size=5000, anomalies=anomalies, anomaly_p=0.4, num_attr=[1, 2, 3, 4],
                               seed=1337)
    generate_for_process_model(process_model, size=5000, anomalies=anomalies,anomaly_p=0.45, num_attr=[1, 2, 3, 4], seed=1337)


