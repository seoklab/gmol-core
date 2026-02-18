import logging
from dataclasses import dataclass

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from rdkit import Chem

from gmol.base.wrapper.rdkit import smi2mol
from .assembly import Branch, ResidueId
from .parse import ChemComp, ChemCompAtom, ChemCompBond

_logger = logging.getLogger(__name__)


@dataclass
class RefLigandInput:
    smiles: str
    atom_ids: list[str]


@dataclass
class MmcifRefLigand:
    atoms: list[ChemCompAtom]
    bonds: list[ChemCompBond]


def mol_from_chem_comp(
    atoms: list[ChemCompAtom], bonds: list[ChemCompBond]
) -> Chem.Mol:
    """Return a RDKit molecule from a list of atoms and bonds.

    :param atoms: list of atoms.
    :param bonds: list of bonds.
    :returns: A molecule with correct atom id (``GetProp("atom_id")``,
        stereochemistry (``GetProp("absolute_config")``) and aromaticity. All
        hydrogens are removed.
    """

    rw_mol = Chem.RWMol()

    bond_ids = {b.atom_id_1 for b in bonds} | {b.atom_id_2 for b in bonds}

    atom_id_to_index: dict[str, int] = {}
    for atom_data in atoms:
        # Some entries produce isolated explicit H atoms during processing.
        # RemoveHs() cannot delete degree-0 H atoms and emits warnings, so
        # drop them here when the bond/heavy-atom partner is missing.
        # Ref: seoklab/gmol-base#9.
        if atom_data.type_symbol == "H" and atom_data.atom_id not in bond_ids:
            continue

        atom = Chem.Atom(atom_data.type_symbol)
        atom.SetProp("atom_id", atom_data.atom_id)
        atom.SetIsAromatic(atom_data.pdbx_aromatic_flag)
        atom.SetFormalCharge(int(atom_data.charge))

        if atom_data.pdbx_stereo_config is not None:
            atom.SetProp("absolute_config", atom_data.pdbx_stereo_config)
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)

        atom_idx = rw_mol.AddAtom(atom)
        atom_id_to_index[atom_data.atom_id] = atom_idx

    for bond_data in bonds:
        i = atom_id_to_index[bond_data.atom_id_1]
        j = atom_id_to_index[bond_data.atom_id_2]
        bond_order = {
            1: Chem.BondType.SINGLE,
            2: Chem.BondType.DOUBLE,
            3: Chem.BondType.TRIPLE,
            4: Chem.BondType.QUADRUPLE,
        }.get(bond_data.value_order, Chem.BondType.SINGLE)
        bond_cnt = rw_mol.AddBond(i, j, bond_order)

        bond = rw_mol.GetBondWithIdx(bond_cnt - 1)
        bond.SetIsAromatic(bond_data.pdbx_aromatic_flag)
        if bond_data.pdbx_stereo_config is not None:
            bond.SetProp("absolute_config", bond_data.pdbx_stereo_config)

    mol = rw_mol.GetMol()
    Chem.SanitizeMol(mol)
    mol = Chem.RemoveHs(mol)

    # This must be done here because hydrogens could be removed
    bond: Chem.Bond
    for bond in mol.GetBonds():
        if not bond.HasProp("absolute_config"):
            continue
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue

        # Skip stereo bonds with insufficient substituent neighbors
        a = bond.GetBeginAtom()
        b = bond.GetEndAtom()
        a_subs = [
            n.GetIdx() for n in a.GetNeighbors() if n.GetIdx() != b.GetIdx()
        ]
        b_subs = [
            n.GetIdx() for n in b.GetNeighbors() if n.GetIdx() != a.GetIdx()
        ]
        if not a_subs or not b_subs:
            _logger.warning(
                (
                    "Invalid bond stereo config: bond %d (%d-%d) has "
                    "absolute_config but insufficient neighbors"
                ),
                bond.GetIdx(),
                a.GetIdx(),
                b.GetIdx(),
            )
            continue

        bond.SetStereo(Chem.BondStereo.STEREOE)
        # Required, set which neighbor is the "stereo atom".
        bond.SetStereoAtoms(a_subs[0], b_subs[0])

    # Required, if not specified, resulting smiles omits E/Z stereochemistry
    Chem.AssignStereochemistry(mol, force=True)

    chiral_atoms: list[int] = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.HasProp("absolute_config")
    ]
    configured_bonds: list[int] = [
        bond.GetIdx()
        for bond in mol.GetBonds()
        if bond.HasProp("absolute_config")
    ]
    Chem.AssignCIPLabels(mol, chiral_atoms, configured_bonds)

    for i in chiral_atoms:
        atom = mol.GetAtomWithIdx(i)
        # Fixed this line
        if atom.HasProp("_CIPCode") and atom.HasProp("absolute_config"):
            if atom.GetProp("_CIPCode") != atom.GetProp("absolute_config"):
                atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)

    for i in configured_bonds:
        bond = mol.GetBondWithIdx(i)
        # Fixed this line
        if bond.HasProp("_CIPCode") and bond.HasProp("absolute_config"):
            if bond.GetProp("_CIPCode") != bond.GetProp("absolute_config"):
                bond.SetStereo(Chem.BondStereo.STEREOZ)

    return mol


def unique_atom_id(residue_id: ResidueId, atom_id: str) -> str:
    return f"{residue_id}-{atom_id}"


def _remap_atom(atom: ChemCompAtom, residue_id: ResidueId) -> ChemCompAtom:
    return ChemCompAtom(
        atom_id=unique_atom_id(residue_id, atom.atom_id),
        type_symbol=atom.type_symbol,
        charge=atom.charge,
        pdbx_aromatic_flag=atom.pdbx_aromatic_flag,
        pdbx_stereo_config=atom.pdbx_stereo_config,
    )


def _remap_bond(bond: ChemCompBond, residue_id: ResidueId) -> ChemCompBond:
    return ChemCompBond(
        atom_id_1=unique_atom_id(residue_id, bond.atom_id_1),
        atom_id_2=unique_atom_id(residue_id, bond.atom_id_2),
        value_order=bond.value_order,
        pdbx_aromatic_flag=bond.pdbx_aromatic_flag,
        pdbx_stereo_config=bond.pdbx_stereo_config,
    )


def _reassign_atom_ids(
    res_cc: list[tuple[ResidueId, ChemComp]],
) -> MmcifRefLigand:
    if len(res_cc) == 1:
        chem_comp = res_cc[0][1]
        return MmcifRefLigand(chem_comp.atoms, chem_comp.bonds)

    ref_ligand = MmcifRefLigand([], [])

    for rid, chem_comp in res_cc:
        ref_ligand.atoms.extend(
            _remap_atom(atom, rid) for atom in chem_comp.atoms
        )
        ref_ligand.bonds.extend(
            _remap_bond(bond, rid) for bond in chem_comp.bonds
        )

    return ref_ligand


def _remove_leaving_atoms(
    ref_ligand: MmcifRefLigand,
    leaving_atoms: set[str],
) -> MmcifRefLigand:
    # Leaving atoms might leave dangling fragments: seoklab/gmol-base#9.

    g = nx.Graph()
    for bond in ref_ligand.bonds:
        g.add_edge(bond.atom_id_1, bond.atom_id_2, bond=bond)
    g.remove_nodes_from(leaving_atoms)

    fragments = list(nx.connected_components(g))
    main = max(fragments, key=len)

    ref_ligand.atoms = [
        atom for atom in ref_ligand.atoms if atom.atom_id in main
    ]
    ref_ligand.bonds = [
        bond for *_, bond in g.subgraph(main).edges(data="bond")
    ]

    return ref_ligand


def _add_inter_residue_bonds(
    ref_ligand: MmcifRefLigand,
    extra_bonds: list[Branch],
) -> MmcifRefLigand:
    if not extra_bonds:
        return ref_ligand

    # Avoid original bonds being modified by reference
    ref_ligand.bonds = ref_ligand.bonds.copy()

    for bond in extra_bonds:
        ref_ligand.bonds.append(
            ChemCompBond(
                atom_id_1=unique_atom_id(
                    bond.ptnr1.res_id, bond.ptnr1.atom_id
                ),
                atom_id_2=unique_atom_id(
                    bond.ptnr2.res_id, bond.ptnr2.atom_id
                ),
                value_order=bond.order,
            )
        )

    leaving_atoms = {
        *(
            unique_atom_id(bond.ptnr1.res_id, bond.ptnr1.leaving_atom_id)
            for bond in extra_bonds
        ),
        *(
            unique_atom_id(bond.ptnr2.res_id, bond.ptnr2.leaving_atom_id)
            for bond in extra_bonds
        ),
    }

    ref_ligand = _remove_leaving_atoms(ref_ligand, leaving_atoms)
    return ref_ligand


def reference_from_mmcif(
    res_cc: list[tuple[ResidueId, ChemComp]],
    extra_bonds: list[Branch] | None = None,
) -> MmcifRefLigand:
    extra_bonds = extra_bonds or []

    ref_ligand = _reassign_atom_ids(res_cc)
    ref_ligand = _add_inter_residue_bonds(ref_ligand, extra_bonds)
    return ref_ligand


def input_from_reference(ref_ligand: MmcifRefLigand) -> RefLigandInput:
    mol = mol_from_chem_comp(ref_ligand.atoms, ref_ligand.bonds)
    smi = Chem.MolToCXSmiles(mol)

    mol = smi2mol(smi)
    atom_ids = [atom.GetProp("atom_id") for atom in mol.GetAtoms()]
    return RefLigandInput(smiles=smi, atom_ids=atom_ids)


def convert_chem_comp(
    ccd: dict[str, ChemComp],
) -> tuple[list[str], dict[str, NDArray[np.str_]]]:
    """Convert CCD to smiles and atom indexes.

    :param ccd: Dictionary of ccd code to ChemComp object.
    """

    all_smiles: list[str] = []
    all_atom_ids: dict[str, NDArray[np.str_]] = {}

    dummy = ResidueId("A", 1, "")
    for ccd_code, chem_comp in ccd.items():
        ref_ligand = reference_from_mmcif([(dummy, chem_comp)])
        ref_input = input_from_reference(ref_ligand)

        all_smiles.append(f"{ref_input.smiles}\t{ccd_code}\n")
        all_atom_ids[ccd_code] = np.array(ref_input.atom_ids, dtype=np.str_)

    return all_smiles, all_atom_ids
