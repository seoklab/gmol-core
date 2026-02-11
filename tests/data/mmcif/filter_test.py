import numpy as np
import pytest

from gmol.core.data.mmcif import Assembly, ChemComp, Mmcif, filter_mmcif


def test_filter_mmcif(
    ccd_components: dict[str, ChemComp],
    sample_assembly: tuple[Mmcif, Assembly],
):
    result = filter_mmcif(*sample_assembly, ccd_components)
    if result is None:
        pytest.skip("target did not pass pre-filter")

    assert not np.isnan(result.coords).any()

    assert "A" in result.chains  # chain A is protein
    assert "B" not in result.chains  # chain B is water molecules
