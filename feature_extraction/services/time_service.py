import datetime
import numpy as np


def calculate_time_features(entry):
    dt = datetime.datetime.strptime(entry["postTimestamp"], '%a %b %d %H:%M:%S %z %Y')
    hours = int(np.floor(dt.timestamp() / 3600))
    weekend = 1
    if 0 <= datetime.date(dt.year, dt.month, dt.day).weekday() <= 4:
        weekend = 0
    return hours, weekend, int(np.round(dt.hour+dt.minute/60))


def get_feat_names():
    return "post_creation_hour", "is_weekend", "hour_of_day"
