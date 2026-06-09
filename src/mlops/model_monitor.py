def monitor_model(mae, rmse):

    if mae > 10:
        status = "POOR"

    elif mae > 5:
        status = "WARNING"

    else:
        status = "GOOD"

    return {
        "status": status,
        "mae": mae,
        "rmse": rmse
    }