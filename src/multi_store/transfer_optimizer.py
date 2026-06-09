def optimize_transfer(
        transfer_units):

    if transfer_units > 0:
        action = "TRANSFER"
    else:
        action = "NO_TRANSFER"

    return {
        "action": action,
        "units": transfer_units
    }