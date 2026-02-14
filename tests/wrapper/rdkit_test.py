from pathlib import Path

import pytest
from rdkit import Chem

from gmol.base.wrapper.rdkit import smi2mol, write_mols


@pytest.fixture
def ethane_mol() -> Chem.Mol:
    return smi2mol("CC")


@pytest.fixture
def benzene_mol() -> Chem.Mol:
    return smi2mol("c1ccccc1")


def test_write_mols_single_mol_sdf(tmp_path: Path, ethane_mol: Chem.Mol):
    out = tmp_path / "out.sdf"
    write_mols(out, ethane_mol)
    assert out.is_file()
    mols = list(Chem.SDMolSupplier(str(out)))
    assert len(mols) == 1
    assert mols[0] is not None
    assert Chem.MolToSmiles(mols[0]) == "CC"


def test_write_mols_single_mol_pdb(tmp_path: Path, ethane_mol: Chem.Mol):
    out = tmp_path / "out.pdb"
    write_mols(out, ethane_mol)
    assert out.is_file()
    content = out.read_text()
    assert "MODEL     1" in content
    assert "ENDMDL" in content


def test_write_mols_list_sdf(
    tmp_path: Path, ethane_mol: Chem.Mol, benzene_mol: Chem.Mol
):
    out = tmp_path / "out.sdf"
    write_mols(out, [ethane_mol, benzene_mol])
    mols = list(Chem.SDMolSupplier(str(out)))
    assert len(mols) == 2
    assert Chem.MolToSmiles(mols[0]) == "CC"
    assert Chem.MolToSmiles(mols[1]) == "c1ccccc1"


def test_write_mols_list_pdb(
    tmp_path: Path, ethane_mol: Chem.Mol, benzene_mol: Chem.Mol
):
    out = tmp_path / "out.pdb"
    write_mols(out, [ethane_mol, benzene_mol])
    content = out.read_text()
    assert "MODEL     1" in content
    assert "MODEL     2" in content
    assert content.count("ENDMDL") == 2


def test_write_mols_skips_none(tmp_path: Path, ethane_mol: Chem.Mol):
    out = tmp_path / "out.sdf"
    write_mols(out, [ethane_mol, None, ethane_mol])
    mols = list(Chem.SDMolSupplier(str(out)))
    assert len(mols) == 2


def test_write_mols_str_path(tmp_path: Path, ethane_mol: Chem.Mol):
    out = tmp_path / "out.sdf"
    write_mols(str(out), ethane_mol)
    assert out.is_file()
    mols = list(Chem.SDMolSupplier(str(out)))
    assert len(mols) == 1


def test_write_mols_unsupported_extension_raises(
    tmp_path: Path, ethane_mol: Chem.Mol
):
    out = tmp_path / "out.xyz"
    with pytest.raises(ValueError, match="Unsupported file extension"):
        write_mols(out, ethane_mol)


def test_write_mols_sdf_kekulize(tmp_path: Path, benzene_mol: Chem.Mol):
    out = tmp_path / "out.sdf"
    write_mols(out, benzene_mol, sdf_kekulize=True)
    assert out.is_file()
    mols = list(Chem.SDMolSupplier(str(out)))
    assert len(mols) == 1
    # Kekulized benzene has explicit single/double bonds
    assert mols[0] is not None
    assert mols[0].GetNumBonds() == 6
