import os
import itertools
from pathlib import Path

import numpy as np
from tqdm import tqdm

from generator.generation.anomaly import *
from generator.generation.attribute_generator import CategoricalAttributeGenerator
from processmining.log import EventLog
from utils.fs import EVENTLOG_DIR

np.random.seed(0)  # This will ensure reproducibility
ps = [0.1]

def get_log_files(path=None):
    # Base
    ROOT_DIR = Path(__file__).parent

    if path is None:
        path = os.path.join(ROOT_DIR/'real-life_Logs') # Randomly generated process models from PLG2
    return [os.path.join(path,f) for f in os.listdir(path)]


logs = [m for m in get_log_files()]
combinations = list(itertools.product(logs, ps))
for event_log_path, p in tqdm(combinations, desc='Add anomalies'):
    print(event_log_path)
    event_log = EventLog.from_xes(event_log_path)
    anomalies = [
        SkipSequenceAnomaly(max_sequence_size=2),
        ReworkAnomaly(max_distance=5, max_sequence_size=3),
        EarlyAnomaly(max_distance=5, max_sequence_size=2),
        LateAnomaly(max_distance=5, max_sequence_size=2),
        InsertAnomaly(max_inserts=2),
    ]

    if event_log.num_event_attributes > 1:
        anomalies.append(AttributeAnomaly(max_events=3, max_attributes=min(2, event_log.num_activities)))

    for anomaly in anomalies:
        # This is necessary to initialize the likelihood graph correctly
        anomaly.activities = event_log.unique_activities
        anomaly.attributes = [CategoricalAttributeGenerator(name=name, values=values) for name, values in
                              event_log.unique_attribute_values.items() if name != 'name']

    for anomaly in anomalies:
        print(anomaly.name)
        for case in tqdm(event_log):
            if np.random.uniform(0, 1) <= p:
                anomaly.apply_to_case(case)
            else:
                NoneAnomaly().apply_to_case(case)
        base_name=os.path.split(event_log_path)[1].split('.')[0]
        event_log.save_json(os.path.join(EVENTLOG_DIR, f'{base_name}-{p:.2f}-{anomaly.name}.json.gz'))
