from pathlib import Path

from gmol.core.data.mmcif import (
    ChemComp,
    ResidueId,
    SequenceToResidue,
    load_components,
    load_mmcif_single,
    mmcif_assemblies,
)


def test_pdb(test_data: Path):
    mmcif = test_data / "mmcif" / "1ubq.cif"
    data = load_mmcif_single(mmcif)

    assert data.entry_id == "1UBQ"


def test_ccd(test_data: Path):
    ccd = test_data / "ccd" / "components_stdres.cif"

    stdres = load_components(ccd)
    assert len(stdres) == 20
    assert list(stdres) == [
        "GLY",
        "ALA",
        "VAL",
        "LEU",
        "ILE",
        "THR",
        "SER",
        "MET",
        "CYS",
        "PRO",
        "PHE",
        "TYR",
        "TRP",
        "HIS",
        "LYS",
        "ARG",
        "ASP",
        "GLU",
        "ASN",
        "GLN",
    ]


def test_load_inconsistent_seq(
    test_data: Path,
    ccd_components: dict[str, ChemComp],
):
    mmcif = test_data / "mmcif" / "3m8z.cif"
    data = load_mmcif_single(mmcif)

    ccd_tmp = ccd_components.copy()
    ccd_tmp["TPO"] = ChemComp.model_validate(
        {
            "id": "TPO",
            "name": "PHOSPHOTHREONINE",
            "type": "L-PEPTIDE LINKING",
            "formula": "C4 H10 N O6 P",
            "formula_weight": 199.099,
            "atoms": [
                {
                    "atom_id": "N",
                    "type_symbol": "N",
                },
                {
                    "atom_id": "CA",
                    "type_symbol": "C",
                },
                {
                    "atom_id": "CB",
                    "type_symbol": "C",
                },
                {
                    "atom_id": "CG2",
                    "type_symbol": "C",
                },
                {
                    "atom_id": "OG1",
                    "type_symbol": "O",
                },
                {
                    "atom_id": "P",
                    "type_symbol": "P",
                },
                {
                    "atom_id": "O1P",
                    "type_symbol": "O",
                },
                {
                    "atom_id": "O2P",
                    "type_symbol": "O",
                },
                {
                    "atom_id": "O3P",
                    "type_symbol": "O",
                },
                {
                    "atom_id": "C",
                    "type_symbol": "C",
                },
                {
                    "atom_id": "O",
                    "type_symbol": "O",
                },
                {
                    "atom_id": "OXT",
                    "type_symbol": "O",
                    "pdbx_leaving_atom_flag": True,
                },
            ],
            "bonds": [
                {
                    "atom_id_1": "N",
                    "atom_id_2": "CA",
                    "value_order": 1,
                },
                {
                    "atom_id_1": "CA",
                    "atom_id_2": "CB",
                    "value_order": 1,
                },
                {
                    "atom_id_1": "CA",
                    "atom_id_2": "C",
                    "value_order": 1,
                },
                {
                    "atom_id_1": "CB",
                    "atom_id_2": "CG2",
                    "value_order": 1,
                },
                {
                    "atom_id_1": "CB",
                    "atom_id_2": "OG1",
                    "value_order": 1,
                },
                {
                    "atom_id_1": "OG1",
                    "atom_id_2": "P",
                    "value_order": 1,
                },
                {
                    "atom_id_1": "P",
                    "atom_id_2": "O1P",
                    "value_order": 2,
                },
                {
                    "atom_id_1": "P",
                    "atom_id_2": "O2P",
                    "value_order": 1,
                },
                {
                    "atom_id_1": "P",
                    "atom_id_2": "O3P",
                    "value_order": 1,
                },
                {
                    "atom_id_1": "C",
                    "atom_id_2": "O",
                    "value_order": 2,
                },
                {
                    "atom_id_1": "C",
                    "atom_id_2": "OXT",
                    "value_order": 1,
                },
            ],
        }
    )

    assemblies = mmcif_assemblies(data, ccd_tmp)
    assert len(assemblies) == 3
    asm = assemblies[0]

    res_id = ResidueId("A", 85)

    for atom in asm.atoms_of_residue(asm.residues[res_id]):
        assert atom.comp_id == "THR"

    het_chains: list[SequenceToResidue] = [
        s2r for s2r in asm.chains["A"].seqres if s2r.res_id == res_id
    ]
    assert len(het_chains) == 1
    assert het_chains[0].comp_id == "THR"
