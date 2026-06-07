from src.simulation.digital_twin import run_digital_twin
from src.simulation.what_if_simulator import run_what_if
from src.simulation.market_shock_simulator import simulate_market_shock


def run_simulation_agent(
        inventory,
        forecast,
        price):

    twin = run_digital_twin(
        inventory,
        forecast,
        20
    )

    what_if = run_what_if(
        forecast,
        price,
        10
    )

    shock = simulate_market_shock(
        forecast,
        10
    )

    return {
        "digital_twin": twin,
        "what_if": what_if,
        "market_shock": shock
    }