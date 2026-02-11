from pathlib import Path

import pytest
from rdkit import Chem

from gmol.core.data.mmcif import ChemComp, ResidueId, load_components
from gmol.core.data.mmcif.smiles import (
    input_from_reference,
    mol_from_chem_comp,
    reference_from_mmcif,
)


@pytest.fixture(scope="session")
def chem_comp(test_data: Path):
    ccd_file = test_data / "ccd" / "components_0IE.cif"
    return load_components(ccd_file)["0IE"]


def test_mol_from_chem_comp(chem_comp: ChemComp):
    mol = mol_from_chem_comp(chem_comp.atoms, chem_comp.bonds)

    heavy_atoms = {
        atom.atom_id: atom
        for atom in chem_comp.atoms
        if atom.type_symbol != "H"
    }
    bonds = {
        tuple(sorted([bond.atom_id_1, bond.atom_id_2])): bond
        for bond in chem_comp.bonds
    }

    atom: Chem.Atom
    for atom in mol.GetAtoms():
        atom_id = atom.GetProp("atom_id")
        assert atom_id in heavy_atoms
        adata = heavy_atoms[atom_id]

        assert atom.GetAtomicNum() > 1
        assert atom.GetSymbol() == adata.type_symbol
        assert atom.GetIsAromatic() == adata.pdbx_aromatic_flag
        if adata.pdbx_stereo_config is not None:
            assert atom.GetProp("absolute_config") == adata.pdbx_stereo_config
            assert (
                atom.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CW
                or atom.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CCW
            )

    bond: Chem.Bond
    for bond in mol.GetBonds():
        bond_id = tuple(
            sorted(
                [
                    bond.GetBeginAtom().GetProp("atom_id"),
                    bond.GetEndAtom().GetProp("atom_id"),
                ]
            )
        )
        assert bond_id in bonds

        bdata = bonds[bond_id]
        if bond.GetIsAromatic():
            assert bdata.pdbx_aromatic_flag
        else:
            assert not bdata.pdbx_aromatic_flag
            assert (
                bond.GetBondType()
                == {
                    1: Chem.BondType.SINGLE,
                    2: Chem.BondType.DOUBLE,
                    3: Chem.BondType.TRIPLE,
                    4: Chem.BondType.QUADRUPLE,
                }[bdata.value_order]
            )

        if bdata.pdbx_stereo_config is not None:
            assert bond.GetProp("absolute_config") == bdata.pdbx_stereo_config


def test_to_smiles(chem_comp: ChemComp):
    ref_input = reference_from_mmcif([(ResidueId("A", 1, ""), chem_comp)])
    ref = input_from_reference(ref_input)
    assert Chem.MolFromSmiles(ref.smiles) is not None
    assert set(ref.atom_ids) == {
        atom.atom_id for atom in chem_comp.atoms if atom.type_symbol != "H"
    }
