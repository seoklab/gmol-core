from pathlib import Path

import pytest
from rdkit import Chem

from gmol.base.wrapper.rdkit import (
    generate_conformer,
    read_mols,
    smi2mol,
    write_mols,
)


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


def test_write_mols_single_mol_multi_conf_sdf(
    tmp_path: Path,
    ethane_mol: Chem.Mol,
):
    mol = generate_conformer(ethane_mol, ignore_failures=False)

    out = tmp_path / "out.sdf"
    write_mols(out, mol)
    assert out.is_file()
    mols = list(Chem.SDMolSupplier(str(out)))
    assert len(mols) == 1
    assert mols[0] is not None
    assert Chem.MolToSmiles(mols[0]) == "CC"

    mol.AddConformer(mol.GetConformer(), assignId=True)

    write_mols(out, mol)
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
    assert "MODEL        1" in content
    assert "ENDMDL" in content


def test_write_mols_single_mol_multi_conf_pdb(
    tmp_path: Path,
    ethane_mol: Chem.Mol,
):
    mol = generate_conformer(ethane_mol, ignore_failures=False)

    out = tmp_path / "out.pdb"
    write_mols(out, mol)
    assert out.is_file()
    content = out.read_text()
    assert content.count("MODEL        1") == 1
    assert content.count("MODEL") == 1
    assert content.count("ENDMDL") == 1

    mol.AddConformer(mol.GetConformer(), assignId=True)

    write_mols(out, mol)
    assert out.is_file()
    content = out.read_text()
    assert content.count("MODEL        1") == 1
    assert content.count("MODEL") == 1
    assert content.count("ENDMDL") == 1


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
    assert "MODEL        1" in content
    assert "MODEL        2" in content
    assert content.count("ENDMDL") == 2
    assert content.count("END") == 3  # 2 ENDMDL + 1 final END


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


LIGAND_NAMES = ["5dwr_ligand", "8a0b_ligand"]


@pytest.mark.parametrize("ligand", LIGAND_NAMES)
def test_read_mols_sanitize(test_data: Path, ligand: str):
    """read_mols with sanitize=True and sanitize=False both return mols."""
    sdf_path = test_data / "ligands" / f"{ligand}.sdf"
    mols_sanitize = read_mols(sdf_path, sanitize=True)
    mols_no_sanitize = read_mols(sdf_path, sanitize=False)
    assert len(mols_sanitize) == len(mols_no_sanitize)
    assert len(mols_sanitize) >= 1
    for a, b in zip(mols_sanitize, mols_no_sanitize, strict=True):
        assert a.GetNumHeavyAtoms() == b.GetNumHeavyAtoms()


@pytest.mark.parametrize("ligand", LIGAND_NAMES)
def test_read_mols_remove_h(test_data: Path, ligand: str):
    """read_mols with remove_h=True returns mols with no more atoms than remove_h=False."""
    sdf_path = test_data / "ligands" / f"{ligand}.sdf"
    mols_with_h = read_mols(sdf_path, remove_h=False)
    mols_no_h = read_mols(sdf_path, remove_h=True)
    assert len(mols_with_h) == len(mols_no_h)
    assert len(mols_with_h) >= 1
    for with_h, no_h in zip(mols_with_h, mols_no_h, strict=True):
        assert no_h.GetNumAtoms() <= with_h.GetNumAtoms()


def test_read_mols_multiple_mol2(test_data: Path):
    mol2_path = test_data / "ligands" / "ligands.mol2"
    mols = read_mols(mol2_path)
    assert len(mols) == 2

    assert mols[0].GetNumAtoms() == 56
    assert mols[0].GetNumBonds() == 59

    assert mols[1].GetNumAtoms() == 22
    assert mols[1].GetNumBonds() == 25


def test_read_mols_multiple_pdb(test_data: Path):
    pdb_path = test_data / "ligands" / "ligands.pdb"
    mols = read_mols(pdb_path)
    assert len(mols) == 2

    assert mols[0].GetNumAtoms() == 56
    assert mols[0].GetNumBonds() == 59

    assert mols[1].GetNumAtoms() == 22
    assert mols[1].GetNumBonds() == 25


@pytest.mark.parametrize("ligand", LIGAND_NAMES)
def test_write_mols_ligands_sdf(tmp_path: Path, test_data: Path, ligand: str):
    """Read SDF from ligands dir, write to tmp SDF, read back and compare count."""
    sdf_path = test_data / "ligands" / f"{ligand}.sdf"
    mols = read_mols(sdf_path)
    assert len(mols) >= 1

    out = tmp_path / "out.sdf"
    write_mols(out, mols)
    assert out.is_file()
    roundtrip = list(Chem.SDMolSupplier(str(out)))
    roundtrip = [m for m in roundtrip if m is not None]
    assert len(roundtrip) == len(mols)
    for a, b in zip(mols, roundtrip, strict=True):
        assert a.GetNumHeavyAtoms() == b.GetNumHeavyAtoms()


@pytest.mark.parametrize("ligand", LIGAND_NAMES)
def test_write_mols_ligands_multi_sdf(
    tmp_path: Path, test_data: Path, ligand: str
):
    """Read multi-molecule SDF from ligands dir, write and roundtrip."""
    sdf_path = test_data / "ligands" / f"{ligand}.sdf"
    mols = read_mols(sdf_path)
    assert len(mols) >= 1

    out = tmp_path / "out.sdf"
    multi_confs = mols * 2
    write_mols(out, multi_confs)
    assert out.is_file()
    roundtrip = list(Chem.SDMolSupplier(str(out)))
    roundtrip = [m for m in roundtrip if m is not None]
    assert len(roundtrip) == len(multi_confs)


@pytest.mark.parametrize("ligand", LIGAND_NAMES)
def test_write_mols_ligands_pdb_roundtrip(
    tmp_path: Path, test_data: Path, ligand: str
):
    """Read PDB from ligands dir, write to tmp PDB, check structure and model count."""
    pdb_path = test_data / "ligands" / f"{ligand}.pdb"
    mols = read_mols(pdb_path)
    assert len(mols) >= 1

    out = tmp_path / "out.pdb"
    write_mols(out, mols)
    assert out.is_file()
    content = out.read_text()
    assert "MODEL        1" in content
    assert "ENDMDL" in content
    assert content.count("MODEL") == len(mols)
    assert content.count("ENDMDL") == len(mols)


@pytest.mark.parametrize("ligand", LIGAND_NAMES)
def test_write_mols_ligands_from_mol2_to_sdf(
    tmp_path: Path, test_data: Path, ligand: str
):
    """Read MOL2 from ligands dir, write as SDF, verify mol count."""
    mol2_path = test_data / "ligands" / f"{ligand}.mol2"
    mols = read_mols(mol2_path)
    assert len(mols) >= 1

    out = tmp_path / "out.sdf"
    write_mols(out, mols)
    assert out.is_file()
    written = list(Chem.SDMolSupplier(str(out)))
    written = [m for m in written if m is not None]
    assert len(written) == len(mols)


@pytest.mark.parametrize("ligand", LIGAND_NAMES)
def test_write_mols_ligands_from_mol2_to_pdb(
    tmp_path: Path, test_data: Path, ligand: str
):
    """Read MOL2 from ligands dir, write as PDB, verify model count."""
    mol2_path = test_data / "ligands" / f"{ligand}.mol2"
    mols = read_mols(mol2_path)
    assert len(mols) >= 1

    out = tmp_path / "out.pdb"
    write_mols(out, mols)
    assert out.is_file()
    content = out.read_text()
    assert content.count("MODEL") == len(mols)
    assert content.count("ENDMDL") == len(mols)


@pytest.mark.parametrize("ligand", LIGAND_NAMES)
def test_write_mols_ligands_skips_none(
    tmp_path: Path, test_data: Path, ligand: str
):
    """write_mols skips None when list is built from ligand data."""
    sdf_path = test_data / "ligands" / f"{ligand}.sdf"
    mols = read_mols(sdf_path)
    assert len(mols) >= 1
    mixed = [mols[0], None, mols[0]]

    out = tmp_path / "out.sdf"
    write_mols(out, mixed)
    written = list(Chem.SDMolSupplier(str(out)))
    written = [m for m in written if m is not None]
    assert len(written) == 2
