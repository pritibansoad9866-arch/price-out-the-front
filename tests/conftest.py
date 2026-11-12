import pytest

from potf.pipeline import run
from potf.simulate import simulate


@pytest.fixture(scope="session")
def sim_run():
    p = simulate(n=900, days=110, seed=3)
    return run(p, seed=3)


@pytest.fixture(scope="session")
def honest_run():
    p = simulate(n=900, days=110, seed=5, front_rate=0.0, syndicate_size=0)
    return run(p, seed=5)
