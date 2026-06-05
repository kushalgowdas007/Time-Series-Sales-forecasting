import pandas as pd


def detect_shift(series):

    historical_mean = series[:-5].mean()

    recent_mean = series[-5:].mean()

    shift_ratio = recent_mean / historical_mean

    if shift_ratio > 1.5:
        return {
            "shift_detected": True,
            "ratio": round(shift_ratio, 2)
        }

    return {
        "shift_detected": False,
        "ratio": round(shift_ratio, 2)
    }


if __name__ == "__main__":

    data = pd.Series([
        100,105,102,108,110,
        220,240,250,245,260
    ])

    print(detect_shift(data))