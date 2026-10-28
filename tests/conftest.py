import os
import sys
import warnings

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("USE_MOCK", "1")
warnings.filterwarnings("ignore", category=FutureWarning)


@pytest.fixture(scope="session")
def panel():
    from data import generate_panel
    return generate_panel()


@pytest.fixture(scope="session")
def small_panel():
    from data import generate_panel
    return generate_panel(n_tickers=20, years=3, n_delisted=4, n_ipo=3)
