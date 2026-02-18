import logging
import re
from collections.abc import Iterable
from itertools import count
from pathlib import Path
from typing import cast

from rdkit import Chem
from rdkit.Chem import AllChem

_logger = logging.getLogger(__name__)


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
        params.maxIterations = max_attempts
        success = False

        if AllChem.EmbedMolecule(new_m, params) == 0:
            success = True

        if not success:
            _logger.warning(
                "Initial embedding failed, retrying with random coordinates."
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
            _logger.warning(
                "MMFF optimization returned status %s.", opt_status
            )

        # Assign new stereochemistry for featurizers
        result = Chem.RemoveHs(new_m)
        Chem.rdmolops.AssignStereochemistryFrom3D(result)
        return result

    except Exception:
        if ignore_failures:
            _logger.exception(
                "Generating conformer failed. Use the original conformer."
            )
            return orig_m

        raise


def _split_read_mol2(path: str):

    def _read_mol(block: list[str]):
        mol2 = "".join(block)
        mol = Chem.MolFromMol2Block(
            mol2,
            sanitize=False,
            removeHs=False,
        )
        return cast(Chem.Mol | None, mol)

    with open(path) as f:
        block: list[str] = []

        for line in f:
            if line.startswith("@<TRIPOS>MOLECULE") and block:
                mol = _read_mol(block)
                if mol is not None:
                    yield mol

                block.clear()

            block.append(line)

        if block:
            mol = _read_mol(block)
            if mol is not None:
                yield mol


def read_mols(
    file_path: Path | str,
    sanitize: bool = True,
    remove_h: bool = False,
) -> list[Chem.Mol]:
    """Read a molecular file into a list of RDKit Mol objects.

    Supported formats: ``.mol2``, ``.sdf``, ``.pdb``.

    :param file_path: Path to the molecular file.
    :param sanitize: If True, sanitize each molecule after reading.
    :param remove_h: If True, remove explicit hydrogens from each molecule.
    :returns: List of RDKit Mol objects; entries that failed to parse are omitted.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = file_path.suffix.lower()
    if ext not in (".mol2", ".sdf", ".pdb"):
        raise ValueError(
            f"Unsupported file extension '{ext}'. Supported formats: .mol2, .sdf, .pdb."
        )

    if ext == ".mol2":
        mols = list(_split_read_mol2(str(file_path)))
    elif ext == ".sdf":
        with Chem.SDMolSupplier(
            str(file_path), sanitize=False, removeHs=False
        ) as suppl:
            mols = [m for m in suppl if m is not None]  # pyright: ignore[reportUnnecessaryComparison]
    elif ext == ".pdb":
        mol = cast(
            Chem.Mol | None,
            Chem.MolFromPDBFile(
                str(file_path), sanitize=False, removeHs=False
            ),
        )
        mols = [mol] if mol is not None else []

    if sanitize:
        for mol in mols:
            Chem.SanitizeMol(mol)

    if remove_h:
        mols = [Chem.RemoveHs(mol, sanitize=sanitize) for mol in mols]

    return mols


_end_re = re.compile(r"^END\s+", re.MULTILINE)


def write_mols(
    save_path: Path | str,
    mols: Chem.Mol | None | Iterable[Chem.Mol | None],
    sdf_kekulize: bool = False,
) -> None:
    """Write a list of RDKit Mol objects to a file.

    Supported extensions are ``.sdf`` and ``.pdb``.

    :param save_path: Path to the output file.
    :param mols: Molecule or list of molecules to write; None entries are skipped.
    :param sdf_kekulize: If True, kekulize molecules when writing to an SDF file.
    """
    if mols is None:
        return

    if isinstance(mols, Chem.Mol):
        mols = [mols]

    save_path = Path(save_path)
    ext = save_path.suffix.lower()

    if ext == ".sdf":
        with Chem.SDWriter(str(save_path)) as w:
            w.SetKekulize(sdf_kekulize)
            for m in mols:
                if m is not None:
                    w.write(m)
    elif ext == ".pdb":
        with save_path.open("w") as f:
            written = False
            model_num = count(1)
            for m in mols:
                if m is not None:
                    written = True
                    idx = next(model_num)

                    # rdkit writes all conformers to pdb by default, but we
                    # only want to write single conformer per molecule.
                    # If there are no conformers, confId=-1 will be used to
                    # write the molecule without 3D coordinates.
                    confid = -1
                    if m.GetNumConformers() > 1:
                        confid = m.GetConformer().GetId()

                    pdb_block = Chem.MolToPDBBlock(m, confId=confid)
                    pdb_block = _end_re.sub("", pdb_block)
                    f.write(f"MODEL {idx:8d}\n")
                    f.write(pdb_block)
                    f.write("ENDMDL\n")

            if written:
                f.write("END\n")
    else:
        raise ValueError(
            f"Unsupported file extension '{ext}'. Supported: .sdf, .pdb"
        )
