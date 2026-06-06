import numpy as np


def forecast_confidence(actual, predicted):

    actual = np.array(actual)

    predicted = np.array(predicted)

    error = np.mean(
        np.abs(actual - predicted)
    )

    confidence = max(
        0,
        100 - error
    )

    return {
        "confidence": round(confidence, 2)
    }


if __name__ == "__main__":

    actual = [30,35,32,36,40]

    predicted = [31,34,33,35,39]

    print(
        forecast_confidence(
            actual,
            predicted
        )
    )