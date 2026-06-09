from src.mlops.model_monitor import monitor_model
from src.mlops.drift_detector import detect_drift
from src.mlops.retraining_agent import retraining_decision

monitor = monitor_model(
    mae=4.2,
    rmse=5.8
)

print("\nMODEL MONITOR")
print(monitor)

drift = detect_drift(
    historical_sales=[
        30,32,35,31,33
    ],
    recent_sales=[
        70,75,80,72,78
    ]
)

print("\nDRIFT DETECTOR")
print(drift)

decision = retraining_decision(
    drift["drift_detected"]
)

print("\nRETRAINING AGENT")
print(decision)