import numpy as np

def detect_drift(old_data, new_data):

    old_mean = np.mean(old_data)
    new_mean = np.mean(new_data)

    drift = abs(new_mean - old_mean)

    return {
        "drift_score": round(drift, 2),
        "drift_detected": drift > 20
    }