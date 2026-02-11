from pathlib import Path

import pytest

from gmol.core.data.mmcif import (
    ChemComp,
    load_components,
    load_mmcif_single,
    mmcif_assemblies,
)


@pytest.fixture(scope="session")
def ccd_components(test_data: Path):
    ccd_file = test_data / "ccd" / "components_stdres.cif"
    return load_components(ccd_file)


@pytest.fixture(scope="session")
def sample_assembly(test_data: Path, ccd_components: dict[str, ChemComp]):
    mmcif = test_data / "mmcif" / "1ubq.cif"
    data = load_mmcif_single(mmcif)
    assemblies = mmcif_assemblies(data, ccd_components)
    assert len(assemblies) == 1
    return data, assemblies[0]
