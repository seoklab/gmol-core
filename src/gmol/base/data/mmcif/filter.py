import datetime as dt

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.spatial import KDTree

from gmol.base.const import (
    CRYSTALLIZATION_AIDS,
    CRYSTALLOGRAPHY_METHODS,
    IONS,
    LIGAND_EXCLUSION_LIST,
    aa_restype_3to1,
)
from .assembly import Assembly, AssemblyAtom, MolType, Residue
from .parse import ChemComp

__all__ = ["filter_mmcif", "filter_structure"]


def new_atom_mask(assembly: Assembly) -> NDArray[np.bool_]:
    return np.zeros(len(assembly.atoms), dtype=bool)


def remove_min_resolved_chains(
    assembly: Assembly,
    min_resolved: int = 4,
):
    mask = new_atom_mask(assembly)

    for chain in assembly.chains_of_type(MolType.Protein):
        if len(chain.residue_ids) < min_resolved:
            mask[chain.atom_idxs] = True
    return mask


def remove_waters(assembly: Assembly) -> NDArray[np.bool_]:
    return np.isin(assembly.comp_ids, ["HOH", "DOD", "DOH"])


def remove_all_unknown_chains(assembly: Assembly):
    mask = new_atom_mask(assembly)
    for chain in assembly.chains_of_type(MolType.Protein):
        if all(
            residue.chem_comp.id not in aa_restype_3to1
            for residue in assembly.residues_of_chain(chain)
        ):
            mask[chain.atom_idxs] = True
    return mask


def _find_clashes(
    coords: NDArray[np.float64], cutoff: float = 1.7
) -> NDArray[np.int64]:
    kdtree = KDTree(coords)
    neighbors = kdtree.query_pairs(cutoff, output_type="ndarray")
    return neighbors


def _exclude_metals(
    atom_comps: NDArray[np.str_],
    clashes: NDArray[np.int64],
):
    clashing_comps = atom_comps[clashes]
    has_metal = np.isin(clashing_comps, list(IONS))
    return clashes[~has_metal.any(axis=1)]


def _per_chain_clashes_ratio(
    atoms: list[AssemblyAtom],
    chain_lengths: pd.Series,
    clashes: NDArray[np.int64],
) -> pd.DataFrame:
    atom_chains = np.array([atom.chain_id for atom in atoms], dtype=np.str_)
    clashing_chains = atom_chains[clashes]
    clashing_chains = clashing_chains[
        clashing_chains[:, 0] != clashing_chains[:, 1]
    ]

    df = pd.DataFrame(clashing_chains)
    df = df.groupby(by=0, sort=False, as_index=False).value_counts(
        sort=False, dropna=False
    )
    df = pd.concat([df, df.rename(columns={0: 1, 1: 0})], axis=0).set_index(0)
    df["ratio"] = df["count"] / chain_lengths.loc[df.index].array
    return df


def _chain_to_remove(
    df: pd.DataFrame,
    chain_lengths: pd.Series,
    cutoff: float,
) -> str | None:
    high_clash_cnts = (df["ratio"] > cutoff).groupby(level=0, sort=False).sum()

    max_clash_cnt = high_clash_cnts.max()
    if not max_clash_cnt:
        return None

    candidates = high_clash_cnts[high_clash_cnts == max_clash_cnt].index
    if len(candidates) == 1:
        return candidates[0]

    candidate_stats = df[df.index.isin(candidates)]
    ratio_sum = candidate_stats["ratio"].groupby(level=0, sort=False).sum()

    candidates = ratio_sum[ratio_sum >= ratio_sum.max() - 1e-3].index
    if len(candidates) == 1:
        return candidates[0]

    lengths = chain_lengths.loc[candidates].sort_index(ascending=False)
    return lengths.idxmin()


def remove_clashing_chains(
    assembly: Assembly,
    clash_threshold: float = 1.7,
    clash_cutoff: float = 0.3,
):
    mask = new_atom_mask(assembly)

    clashes = _find_clashes(assembly.coords, clash_threshold)
    clashes = _exclude_metals(assembly.comp_ids, clashes)

    ch_lengths = pd.Series(
        [len(ch.atom_idxs) for ch in assembly.chains.values()],
        index=list(assembly.chains.keys()),
    )

    df = _per_chain_clashes_ratio(assembly.atoms, ch_lengths, clashes)
    while True:
        if df.empty:
            break

        ch = _chain_to_remove(df, ch_lengths, clash_cutoff)
        if ch is None:
            break

        mask[assembly.chains[ch].atom_idxs] = True
        df = df[(df.index != ch) & (df[1] != ch)]

    return mask


def remove_non_ccd_atoms(
    assembly: Assembly,
    chem_comp_dict: dict[str, ChemComp],
):
    mask = new_atom_mask(assembly)

    removed_atoms: list[int] = []
    for chain in assembly.chains.values():
        for residue in assembly.residues_of_chain(chain):
            result = chem_comp_dict.get(residue.chem_comp.id, None)
            if result is None:
                continue

            ccd_atoms: set[str] = set([atom.atom_id for atom in result.atoms])
            removed_atoms.extend(
                atom.atom_idx
                for atom in assembly.atoms_of_residue(residue)
                if atom.atom_id not in ccd_atoms
            )

    mask[removed_atoms] = True
    return mask


def remove_large_ca_distance(
    assembly: Assembly,
    max_distance: float = 10,
):
    mask = new_atom_mask(assembly)

    for chain in assembly.chains_of_type(MolType.Protein):
        chain_ca_coords = np.full(
            (len(chain.seqres), 3), np.nan, dtype=np.float64
        )

        indices: list[int] = []
        known_ca_coords: list[NDArray[np.float64]] = []
        for i, seqres in enumerate(chain.seqres):
            if seqres.res_id is None:
                continue

            res = assembly.residues[seqres.res_id]
            for atom in assembly.atoms_of_residue(res):
                if atom.atom_id == "CA":
                    indices.append(i)
                    known_ca_coords.append(assembly.coords[atom.atom_idx])
                    break

        if not known_ca_coords:
            continue

        chain_ca_coords[indices] = known_ca_coords

        pairwise_dists = np.linalg.norm(
            chain_ca_coords[1:] - chain_ca_coords[:-1],
            axis=1,
        )
        consec_dists = pairwise_dists[~np.isnan(pairwise_dists)]
        if np.any(consec_dists > max_distance):
            mask[chain.atom_idxs] = True

    return mask


def ligand_atom_mask(assembly: Assembly) -> NDArray[np.bool_]:
    mask = new_atom_mask(assembly)
    for chain in assembly.chains_of_type(MolType.Ligand):
        mask[chain.atom_idxs] = True
    return mask


def remove_crystallization_aids(assembly: Assembly, method: str):
    if method not in CRYSTALLOGRAPHY_METHODS:
        return new_atom_mask(assembly)

    return np.isin(assembly.comp_ids, list(CRYSTALLIZATION_AIDS))


def remove_excluded_ligands(assembly: Assembly):
    return np.isin(assembly.comp_ids, list(LIGAND_EXCLUSION_LIST))


def remove_ions(assembly: Assembly):
    return np.isin(assembly.comp_ids, list(IONS))


def _is_ligand(chem_comp: ChemComp) -> bool:
    return all(
        chem_type not in chem_comp.type
        for chem_type in ["peptide", "rna", "dna"]
    )


def _find_leaving_atoms(
    assembly: Assembly,
    residue: Residue,
    ccd_result: ChemComp,
):
    leaving_atoms_table = {
        cc_atom.atom_id: cc_atom.pdbx_leaving_atom_flag
        for cc_atom in ccd_result.atoms
    }

    leaving_atoms: list[int] = []
    for aid, leaving_flag in leaving_atoms_table.items():
        if not leaving_flag:
            continue

        for atom in assembly.atoms_of_residue(residue):
            if atom.atom_id == aid:
                leaving_atoms.append(atom.atom_idx)

    if len(leaving_atoms) != len(set(leaving_atoms)):
        raise ValueError("Duplicate atom indices found")

    return leaving_atoms


def remove_leaving_atoms(
    assembly: Assembly,
    chem_comp_dict: dict[str, ChemComp],
):
    mask = new_atom_mask(assembly)

    removed_atoms: list[int] = []
    for bond in assembly.connections:
        if bond.conn_type != "covale":
            continue

        if not bond.leaving_atom_count:
            continue

        for atom_idx in (bond.src_idx, bond.dst_idx):
            atom = assembly.atoms[atom_idx]
            if not _is_ligand(assembly.residues[atom.residue_id].chem_comp):
                continue

            ccd_data = chem_comp_dict.get(atom.comp_id, None)
            if ccd_data is None:
                continue

            residue = assembly.residues[atom.residue_id]
            removed_atoms += _find_leaving_atoms(assembly, residue, ccd_data)

    mask[removed_atoms] = True
    return mask


def filter_structure(
    assembly: Assembly,
    exptl_method: str,
    chem_comp_dict: dict[str, ChemComp],
    is_test: bool = False,
) -> NDArray[np.bool_]:
    mask = remove_min_resolved_chains(assembly)
    mask |= remove_waters(assembly)
    mask |= remove_all_unknown_chains(assembly)
    mask |= remove_clashing_chains(assembly)
    mask |= remove_non_ccd_atoms(assembly, chem_comp_dict)
    mask |= remove_large_ca_distance(assembly)
    mask |= remove_leaving_atoms(assembly, chem_comp_dict)
    if is_test:
        lig_mask = ligand_atom_mask(assembly)
        mask |= lig_mask & remove_crystallization_aids(assembly, exptl_method)
        mask |= lig_mask & remove_excluded_ligands(assembly)
    return mask


def filter_mmcif(
    assembly: Assembly,
    chem_comp_dict: dict[str, ChemComp],
    cutoff_date: dt.date = dt.date(2021, 9, 30),
    max_resolution: float = 9.0,
    max_chains: int = 300,
    is_test: bool = False,
) -> Assembly | None:
    if not (
        assembly.metadata.revision_date < cutoff_date
        and assembly.metadata.resolution < max_resolution
        and assembly.count_polymer_chains() <= max_chains
    ):
        return None

    mask = filter_structure(
        assembly,
        assembly.metadata.exptl_method,
        chem_comp_dict,
        is_test,
    )

    return assembly.filter(~mask)
