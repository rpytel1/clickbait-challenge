import datetime

import numpy as np


def calculate_time_features(entry):
    dt = datetime.datetime.strptime(entry["postTimestamp"], '%a %b %d %H:%M:%S %z %Y')

    hours = np.floor(dt.timestamp() / 3600)
    return hours