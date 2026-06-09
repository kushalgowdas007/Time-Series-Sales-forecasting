import numpy as np

def detect_drift(
        historical_sales,
        recent_sales):

    historical_mean = np.mean(
        historical_sales
    )

    recent_mean = np.mean(
        recent_sales
    )

    drift_score = abs(
        recent_mean -
        historical_mean
    )

    return {
        "drift_score":
            round(float(drift_score), 2),

        "drift_detected":
            drift_score > 20
    }