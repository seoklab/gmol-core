from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def test_data() -> Path:
    return Path(__file__).resolve(True).parent / "test_data"
