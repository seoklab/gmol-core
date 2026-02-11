import warnings

from rdkit import Chem
from rdkit.Chem import AllChem


def smi2mol(
    smiles: str,
    kekuilze: bool = False,
    sanitize: bool = True,
) -> Chem.Mol:
    m = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    if m is None:  # pyright: ignore[reportUnnecessaryComparison]
        raise ValueError(f"Fail to convert smiles ({smiles}) to RDKit Mol.")
    if kekuilze:
        Chem.Kekulize(m)
    return m


def mol2smi(mol: Chem.Mol, kekuleSmiles: bool = True) -> str:
    return Chem.MolToSmiles(mol, kekuleSmiles)


def get_all_bonds(mol: Chem.Mol) -> list[tuple[int, int]]:
    return [(t.GetBeginAtomIdx(), t.GetEndAtomIdx()) for t in mol.GetBonds()]


def find_all_bonds_with_atom_idx(
    mol: Chem.Mol, atom_indices: list[int]
) -> list[tuple[int, int]]:
    """
    Find all chemical bonds in the molecule given atom indices.
    """
    all_bonds = get_all_bonds(mol)
    result = set()
    for bond in all_bonds:
        b_src, b_dst = bond
        if b_src in atom_indices and b_dst in atom_indices:
            result.add(tuple(sorted(bond)))
    return list(result)  # type: ignore[arg-type]


def generate_conformer(
    mol: Chem.Mol,
    max_attempts: int = 5,
    optimize_iters: int = 500,
    ignore_failures: bool = True,
) -> Chem.Mol:
    # To preserve original conformer
    orig_m = Chem.Mol(mol)

    try:
        new_m = Chem.Mol(mol)
        new_m.RemoveAllConformers()
        new_m = Chem.AddHs(new_m)

        # Attempt embedding
        params = AllChem.ETKDGv3()
        params.maxAttempts = max_attempts
        success = False

        if AllChem.EmbedMolecule(new_m, params) == 0:
            success = True

        if not success:
            warnings.warn(
                "Initial embedding failed, retrying with random coordinates.",
                stacklevel=1,
            )
            params.useRandomCoords = True
            if AllChem.EmbedMolecule(new_m, params) != 0:
                raise ValueError(
                    "Conformer embedding failed after randomization."
                )

        # MMFF optimization
        opt_status = AllChem.MMFFOptimizeMolecule(
            new_m,
            confId=0,
            mmffVariant="MMFF94s",
            maxIters=optimize_iters,
        )
        if opt_status != 0:
            warnings.warn(
                f"MMFF optimization returned status {opt_status}.",
                stacklevel=1,
            )

        # Assign new stereochemistry for featurizers
        result = Chem.RemoveHs(new_m)
        Chem.rdmolops.AssignStereochemistryFrom3D(result)
        return result

    except Exception as e:
        if ignore_failures:
            warnings.warn(
                "Generating conformer failed. Use the original conformer.",
                stacklevel=1,
            )
            return orig_m
        else:
            raise ValueError(f"Conformer generation failed: {e}") from e
