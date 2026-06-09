def generate_summary(
        forecast,
        revenue,
        risk):

    summary = f"""
Forecast Demand: {forecast}

Expected Revenue: ₹{revenue}

Risk Level: {risk}

Recommendation:
Monitor inventory and demand trends.
"""

    return summary