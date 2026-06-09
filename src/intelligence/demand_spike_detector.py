import pandas as pd


def detect_spike(series):

    historical_avg = series[:-5].mean()

    recent_avg = series[-5:].mean()

    spike_ratio = recent_avg / historical_avg

    if spike_ratio > 1.5:
        return {
            "spike_detected": True,
            "spike_ratio": float(round(spike_ratio, 2))
        }

    return {
        "spike_detected": False,
        "spike_ratio": float(round(spike_ratio, 2))
    }


if __name__ == "__main__":

    sales = pd.Series([
        30,32,31,34,33,
        80,90,100,95,110
    ])

    print(detect_spike(sales))