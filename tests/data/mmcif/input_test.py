import datetime as dt
from pathlib import Path

import numpy as np
import pytest
from pydantic import TypeAdapter

from gmol.core.data.mmcif import (
    ChemComp,
    MolType,
    filter_mmcif,
    load_mmcif_single,
    mmcif_assemblies,
)
from gmol.core.data.mmcif.input import (
    ExtraBondPartner,
    build_input,
    chain_type_to_constants,
)


@pytest.fixture(scope="session")
def ccd(test_data: Path):
    return TypeAdapter(dict[str, ChemComp]).validate_json(
        test_data.joinpath("ccd", "components_min.json").read_bytes()
    )


def _load_input(
    pdb_id: str,
    test_data: Path,
    ccd: dict[str, ChemComp],
    split: bool,
):
    mmcif = test_data / "mmcif" / f"{pdb_id}.cif"
    data = load_mmcif_single(mmcif)

    assemblies = mmcif_assemblies(data, ccd)
    assert len(assemblies) == 1

    result = filter_mmcif(
        data, assemblies[0], ccd, cutoff_date=dt.date(9999, 12, 31)
    )
    assert result is not None

    input_data = build_input(result, data, ccd, split_modified=split)
    return input_data


def test_build_input_basic_transforms(
    test_data: Path,
    ccd: dict[str, ChemComp],
):
    input_data = _load_input("4hf7", test_data, ccd, False)

    # test ARG swap (dist (CD, NH1) < dist(CD, NH2))
    sample_protein = input_data.polymers[0]
    sample_arg_idx = 10
    arg_coords = sample_protein.atom_coords[sample_arg_idx]

    arg_atom_idxs = chain_type_to_constants[MolType.Protein].residue_atom_idxs[
        "ARG"
    ]
    CD_idx, NH1_idx, NH2_idx = (
        arg_atom_idxs["CD"],
        arg_atom_idxs["NH1"],
        arg_atom_idxs["NH2"],
    )

    dist_CD_NH1 = np.linalg.norm(arg_coords[CD_idx] - arg_coords[NH1_idx])
    dist_CD_NH2 = np.linalg.norm(arg_coords[CD_idx] - arg_coords[NH2_idx])

    assert dist_CD_NH1 < dist_CD_NH2, (
        f"ARG residue at index {sample_arg_idx}: "
        f"NH1 ({dist_CD_NH1:.3f}) should be closer to CD than NH2 ({dist_CD_NH2:.3f})."
    )

    # test MSE restype is changed to MET
    sample_mse_idx = 31

    mse_chem_comp = sample_protein.chem_comp_raw[sample_mse_idx]
    assert mse_chem_comp == "MSE", (
        f"MSE residue index {sample_mse_idx} should have chem_comp_id 'MSE'"
    )

    mse_restype = sample_protein.restype[sample_mse_idx]
    met_restype = chain_type_to_constants[
        MolType.Protein
    ].restype_order_with_x["M"]
    assert mse_restype == met_restype, (
        f"MSE residue index {sample_mse_idx} should have restype {met_restype}"
    )

    # test for modified residue: it treat as a covalent-lig (bonded with protein chain)
    modres_chain_id = "A_1-MOD34"
    prev_residue = ("A_1", 33, "C")
    next_residue = ("A_1", 35, "N")

    bond_pairs = {
        frozenset({prev_residue, (modres_chain_id, 0, "N")}),
        frozenset({next_residue, (modres_chain_id, 0, "C")}),
    }
    total_extra_bonds = {
        frozenset(
            {
                (bond.src.chain_id, bond.src.res_idx, bond.src.atom_id),
                (bond.dst.chain_id, bond.dst.res_idx, bond.dst.atom_id),
            }
        )
        for bond in input_data.extra_bonds
    }

    assert bond_pairs.issubset(total_extra_bonds), (
        f"Modified residue ({modres_chain_id}) is connected to previous residue {prev_residue} and next residue {next_residue} correctly."
    )


def test_build_input_equivalent_entity_different_chain(
    test_data: Path,
    ccd: dict[str, ChemComp],
):
    input_data = _load_input("7rtb", test_data, ccd, False)

    modres = [
        lig for lig in input_data.ligands if lig.chain_id.split("-")[0] == "F"
    ]

    modres_entities = {lig.entity_id for lig in modres}
    assert len(modres_entities) == 1, (
        "Same modres should have single entity id"
    )

    modres_chains = {lig.chain_id for lig in modres}
    assert len(modres_chains) == 2, "Same modres should have two chain ids"

    input_data = _load_input("7rtb", test_data, ccd, True)

    modres = [
        lig for lig in input_data.ligands if lig.chain_id.split("-")[0] == "F"
    ]

    modres_entities = {lig.entity_id for lig in modres}
    assert len(modres_entities) == 2, (
        "Must have two splitted side chain entities"
    )

    modres_chains = {lig.chain_id for lig in modres}
    assert len(modres_chains) == 4, "Must have 2 x 2 splitted side chains"


def test_build_input_disulfide_covalent_ligand(
    test_data: Path,
    ccd: dict[str, ChemComp],
):
    input_data = _load_input("7qgw", test_data, ccd, False)

    bonds = {
        tuple(sorted([bond.src, bond.dst])): bond
        for bond in input_data.extra_bonds
    }

    assert (
        ExtraBondPartner(chain_id="A", res_idx=21, atom_id="SG"),
        ExtraBondPartner(chain_id="A", res_idx=64, atom_id="SG"),
    ) in bonds, "Missing disulfide bond between CYS 22.A and CYS 64.A"

    assert (
        ExtraBondPartner(chain_id="A", res_idx=24, atom_id="SG"),
        ExtraBondPartner(chain_id="K", res_idx=0, atom_id="C25"),
    ) in bonds, "Missing covalent bond between CYS 25.A and RN2 309.A"
