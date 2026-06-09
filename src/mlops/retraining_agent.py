def retraining_decision(drift_detected):

    if drift_detected:
        return "RETRAIN_MODEL"

    return "MODEL_OK"