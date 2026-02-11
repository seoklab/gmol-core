import nuri

from gmol.core.data.mmcif import Assembly, Mmcif


def test_to_pdb(sample_assembly: tuple[Mmcif, Assembly]):
    _, assembly = sample_assembly
    result = assembly.to_pdb()

    mols = list(nuri.readstring("pdb", result, sanitize=False))
    assert len(mols) == 1

    mol = mols[0]
    assert len(mol) == len(assembly.atoms)
