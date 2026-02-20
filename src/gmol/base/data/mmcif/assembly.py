import enum
import itertools
from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property
from string import ascii_lowercase, ascii_uppercase, digits
from typing import ClassVar, Protocol

import numpy as np
from numpy.typing import NDArray

from gmol.base.const import (
    aa_restype_3to1,
    dna_restype_3to1,
    modres,
    rna_restype_3to1,
)
from .parse import (
    AtomSite,
    BioAssembly,
    BranchLink,
    BranchLinkPartner,
    ChemComp,
    Entity,
    Mmcif,
    Scheme,
    StructConn,
    SymOp,
)
from .write import mmcif_bond_order, mmcif_bool, mmcif_write_block

__all__ = [
    "Assembly",
    "AssemblyAtom",
    "AssemblyConnection",
    "Chain",
    "MolType",
    "Residue",
    "ResidueId",
    "SequenceToResidue",
    "Transformation",
    "expand_ops",
    "mmcif_assemblies",
    "mmcif_chain_types",
    "polymer_mol_type",
    "residue_mol_type",
]


class MolType(enum.IntEnum):
    Protein = 1
    RNA = 2
    DNA = 3
    Ligand = 4

    @property
    def is_polymer(self) -> bool:
        return self.value < MolType.Ligand


@dataclass(frozen=True, order=True)
class ResidueId:
    chain_id: str
    seq_id: int
    ins_code: str = ""

    def __str__(self):
        return f"{self.chain_id}.{self.seq_id}{self.ins_code}"

    @classmethod
    def from_scheme(cls, scheme: Scheme):
        if scheme.pdb_seq_num is None:
            return None

        return cls(
            scheme.asym_id,
            scheme.pdb_seq_num,
            scheme.pdb_ins_code or "",
        )

    def with_chain_suffix(self, suffix: str):
        return ResidueId(self.chain_id + suffix, self.seq_id, self.ins_code)


@dataclass(order=True)
class Residue:
    residue_id: ResidueId
    chem_comp: ChemComp = field(compare=False)

    atom_idxs: list[int] = field(default_factory=list, compare=False)

    def new_residue(self, atom_idxs: list[int]):
        return Residue(self.residue_id, self.chem_comp, atom_idxs)

    def with_updates(self, offset: int, chain_suffix: str):
        return Residue(
            self.residue_id.with_chain_suffix(chain_suffix),
            self.chem_comp,
            [idx + offset for idx in self.atom_idxs],
        )


@dataclass
class SequenceToResidue:
    comp_id: str
    seq_id: int

    res_id: ResidueId | None

    @classmethod
    def from_scheme(cls, scheme: Scheme):
        return cls(scheme.mon_id, scheme.seq_id, ResidueId.from_scheme(scheme))

    def with_chain_suffix(self, suffix: str):
        return SequenceToResidue(
            self.comp_id,
            self.seq_id,
            self.res_id.with_chain_suffix(suffix) if self.res_id else None,
        )


@dataclass(order=True, frozen=True)
class BranchPartner:
    seq_id: int
    atom_id: str
    leaving_atom_id: str

    comp_id: str = field(compare=False)
    res_id: ResidueId

    @classmethod
    def from_branch_scheme(cls, scheme: Scheme, br_info: BranchLinkPartner):
        res_id = ResidueId.from_scheme(scheme)
        assert res_id is not None

        return cls(
            scheme.seq_id,
            br_info.atom_id,
            br_info.leaving_atom_id,
            scheme.mon_id,
            res_id,
        )

    def with_chain_suffix(self, suffix: str):
        return BranchPartner(
            self.seq_id,
            self.atom_id,
            self.leaving_atom_id,
            self.comp_id,
            self.res_id.with_chain_suffix(suffix),
        )

    def to_scheme(self, entity_id: int) -> Scheme:
        return Scheme(
            asym_id=self.res_id.chain_id,
            entity_id=entity_id,
            mon_id=self.comp_id,
            seq_id=self.seq_id,
            pdb_seq_num=self.res_id.seq_id,
            pdb_ins_code=self.res_id.ins_code or None,
        )


@dataclass(order=True, frozen=True)
class Branch:
    ptnr1: BranchPartner
    ptnr2: BranchPartner
    order: int = field(compare=False)

    def with_chain_suffix(self, suffix: str):
        return Branch(
            self.ptnr1.with_chain_suffix(suffix),
            self.ptnr2.with_chain_suffix(suffix),
            self.order,
        )


@dataclass(order=True)
class Chain:
    chain_id: str

    entity_id: int = field(compare=False)
    type: MolType = field(compare=False)

    seqres: list[SequenceToResidue] = field(
        default_factory=list, compare=False
    )

    branches: list[Branch] = field(default_factory=list, compare=False)

    residue_ids: list[ResidueId] = field(default_factory=list, compare=False)
    atom_idxs: list[int] = field(default_factory=list, compare=False)

    @cached_property
    def resseq(self) -> dict[ResidueId, int]:
        return {
            seq.res_id: i
            for i, seq in enumerate(self.seqres)
            if seq.res_id is not None
        }

    def new_chain(self, residue_ids: list[ResidueId], atom_idxs: list[int]):
        chain = Chain(
            self.chain_id,
            self.entity_id,
            self.type,
            deepcopy(self.seqres),
            deepcopy(self.branches),
            residue_ids,
            atom_idxs,
        )
        chain.sync_mappings()
        return chain

    def with_updates(self, offset: int, suffix: str):
        return Chain(
            self.chain_id + suffix,
            self.entity_id,
            self.type,
            [seqres.with_chain_suffix(suffix) for seqres in self.seqres],
            [branch.with_chain_suffix(suffix) for branch in self.branches],
            [rid.with_chain_suffix(suffix) for rid in self.residue_ids],
            [idx + offset for idx in self.atom_idxs],
        )

    def sync_mappings(self):
        res_ids = set(self.residue_ids)

        for seqres in self.seqres:
            if seqres.res_id is not None and seqres.res_id not in res_ids:
                seqres.res_id = None

        self.branches = [
            branch
            for branch in self.branches
            if branch.ptnr1.res_id in res_ids
            and branch.ptnr2.res_id in res_ids
        ]


@dataclass
class AssemblyAtom:
    atom_idx: int
    residue_id: ResidueId

    type_symbol: str
    atom_id: str
    comp_id: str

    @property
    def chain_id(self) -> str:
        return self.residue_id.chain_id

    def new_atom(self, idx: int):
        return AssemblyAtom(
            atom_idx=idx,
            residue_id=self.residue_id,
            type_symbol=self.type_symbol,
            atom_id=self.atom_id,
            comp_id=self.comp_id,
        )

    def with_updates(self, idx: int, chain_suffix: str):
        return AssemblyAtom(
            atom_idx=idx,
            residue_id=self.residue_id.with_chain_suffix(chain_suffix),
            type_symbol=self.type_symbol,
            atom_id=self.atom_id,
            comp_id=self.comp_id,
        )


class _IndexMapper(Protocol):
    def __getitem__(self, idx: int) -> int: ...


@dataclass
class AssemblyConnection:
    src_idx: int
    dst_idx: int

    conn_type: str
    leaving_atom_count: int

    def remap(self, idx_map: _IndexMapper):
        return AssemblyConnection(
            src_idx=idx_map[self.src_idx],
            dst_idx=idx_map[self.dst_idx],
            conn_type=self.conn_type,
            leaving_atom_count=self.leaving_atom_count,
        )


class Transformation:
    ident: ClassVar["Transformation"] = None  # type: ignore[assignment]

    def __init__(
        self, matrix: NDArray[np.float64], vector: NDArray[np.float64]
    ):
        self.matrix = matrix
        self.vector = vector

    @classmethod
    def from_symop(cls, op: SymOp):
        return cls(op.matrix, op.vector)

    def __add__(self, other: "Transformation") -> "Transformation":
        return Transformation(
            matrix=self.matrix @ other.matrix,
            vector=self.matrix @ other.vector + self.vector,
        )

    def __matmul__(self, coords: NDArray[np.float64]) -> NDArray[np.float64]:
        return (self.matrix @ coords.T + self.vector[:, None]).T


Transformation.ident = Transformation(
    matrix=np.eye(3, dtype=np.float64), vector=np.zeros(3, dtype=np.float64)
)


class _OffsetMapper:
    def __init__(self, offset: int):
        self.offset = offset

    def __getitem__(self, idx: int) -> int:
        return idx + self.offset


@dataclass
class _PdbAtom:
    record: str
    res_id: ResidueId
    res_name: str
    name: str
    coords: NDArray[np.float64]
    element: str

    def __post_init__(self):
        assert len(self.record) == 6
        assert len(self.res_id.chain_id) == 1
        assert -999 <= self.res_id.seq_id <= 9999
        assert len(self.res_id.ins_code) <= 1
        assert len(self.res_name) <= 3
        assert len(self.element) <= 2

    def to_pdb_line(self, serial: int):
        return (
            # 1-21
            f"{self.record}{serial:>5} {self.name} {self.res_name:>3} "
            # 22-27
            f"{self.res_id.chain_id}{self.res_id.seq_id:4d}{self.res_id.ins_code:1}"
            # 28-54
            f"   {self.coords[0]:>8.3f}{self.coords[1]:>8.3f}{self.coords[2]:>8.3f}"
            # 5    6         7
            # 56789012345678901234567
            f"  1.00  0.00          {self.element.upper():>2}  "
        )


class Assembly:
    def __init__(
        self,
        coords: NDArray[np.float64],
        atoms: list[AssemblyAtom],
        residues: dict[ResidueId, Residue],
        chains: dict[str, Chain],
        entities: dict[int, Entity],
        connections: list[AssemblyConnection],
    ):
        self.coords = coords

        self.atoms = atoms
        self.connections = connections

        self.residues = residues
        self.chains = chains
        self.entities = entities

    @cached_property
    def comp_ids(self) -> NDArray[np.str_]:
        return np.array([atom.comp_id for atom in self.atoms], dtype=np.str_)

    @cached_property
    def chem_comps(self) -> dict[str, ChemComp]:
        return {
            res.chem_comp.id: res.chem_comp for res in self.residues.values()
        }

    def count_polymer_chains(self) -> int:
        return len(self.chains) - sum(
            chain.type == MolType.Ligand for chain in self.chains.values()
        )

    def chains_of_type(self, tp: MolType):
        for chain in self.chains.values():
            if chain.type == tp:
                yield chain

    def has_chain_type(self, tp: MolType) -> bool:
        return any(chain.type == tp for chain in self.chains.values())

    def residues_of_chain(self, chain: Chain):
        for rid in chain.residue_ids:
            yield self.residues[rid]

    def atoms_of_chain(self, chain: Chain):
        for aid in chain.atom_idxs:
            yield self.atoms[aid]

    def atoms_of_residue(self, residue: Residue):
        for aid in residue.atom_idxs:
            yield self.atoms[aid]

    def filter(self, selected: NDArray[np.bool_]) -> "Assembly":
        idx_map = np.cumsum(selected.astype(np.int64)) - 1

        coords = self.coords[selected]

        atoms: list[AssemblyAtom] = [
            atom.new_atom(i)
            for i, atom in enumerate(
                np.array(self.atoms, dtype=object)[selected]
            )
        ]

        residues: dict[ResidueId, Residue] = {}
        for rid, residue in self.residues.items():
            atom_idxs = np.array(residue.atom_idxs)
            atom_idxs = idx_map[atom_idxs[selected[atom_idxs]]].tolist()
            if not atom_idxs:
                continue

            residues[rid] = residue.new_residue(atom_idxs)

        chains: dict[str, Chain] = {}
        for cid, chain in self.chains.items():
            atom_idxs = np.array(chain.atom_idxs)
            atom_idxs = idx_map[atom_idxs[selected[atom_idxs]]].tolist()
            if not atom_idxs:
                continue

            chains[cid] = chain.new_chain(
                [rid for rid in chain.residue_ids if rid in residues],
                atom_idxs,
            )

        if self.connections:
            conn_idxs = np.array(
                [(c.src_idx, c.dst_idx) for c in self.connections],
            )
            selected_conns: Iterable[int] = (
                selected[conn_idxs].all(axis=1).nonzero()[0]
            )
            connections = [
                self.connections[i].remap(idx_map) for i in selected_conns
            ]
        else:
            connections = []

        eids = {chain.entity_id for chain in chains.values()}
        entities = {eid: self.entities[eid] for eid in eids}

        return Assembly(
            coords,
            atoms,
            residues,
            chains,
            entities,
            connections,
        )

    def filter_chains(self, chain_ids: list[str]):
        atom_chain_ids = np.array(
            [atom.chain_id for atom in self.atoms], dtype=np.str_
        )
        selected = np.isin(atom_chain_ids, chain_ids)
        return self.filter(selected)

    def transform(self, xform: Transformation) -> "Assembly":
        return Assembly(
            xform @ self.coords,
            self.atoms,
            self.residues,
            self.chains,
            self.entities,
            self.connections,
        )

    def apply(self, operations: list[Transformation]) -> "Assembly":
        return Assembly.join([self.transform(op) for op in operations])

    @classmethod
    def join(cls, assemblies: list["Assembly"]):
        if len(assemblies) == 1:
            return assemblies[0]

        coords = np.concatenate([assembly.coords for assembly in assemblies])

        atoms: list[AssemblyAtom] = []
        residues: dict[ResidueId, Residue] = {}
        chains: dict[str, Chain] = {}
        connections: list[AssemblyConnection] = []

        for i, assembly in enumerate(assemblies, 1):
            suffix = f"_{i}"
            offset = len(atoms)

            atoms.extend(
                atom.with_updates(j, suffix)
                for j, atom in enumerate(assembly.atoms, offset)
            )

            for residue in assembly.residues.values():
                residue = residue.with_updates(offset, suffix)
                residues[residue.residue_id] = residue

            for chain in assembly.chains.values():
                chain = chain.with_updates(offset, suffix)
                chains[chain.chain_id] = chain

            connections.extend(
                [
                    conn.remap(_OffsetMapper(offset))
                    for conn in assembly.connections
                ]
            )

        entities = dict(
            itertools.chain.from_iterable(
                asm.entities.items() for asm in assemblies
            )
        )

        return cls(coords, atoms, residues, chains, entities, connections)

    def to_mmcif(self, name: str, write_schemes: bool = True) -> str:
        merged_branches: dict[int, set[Branch]] = defaultdict(set)
        branch_scheme: dict[str, set[Scheme]] = defaultdict(set)
        nonpoly_scheme: dict[str, set[Scheme]] = defaultdict(set)
        for chain in self.chains.values():
            merged_branches[chain.entity_id].update(chain.branches)

            br_schemes = branch_scheme[chain.chain_id]
            br_schemes.update(
                {
                    branch.ptnr1.to_scheme(chain.entity_id)
                    for branch in chain.branches
                },
                {
                    branch.ptnr2.to_scheme(chain.entity_id)
                    for branch in chain.branches
                },
            )

            if chain.type != MolType.Ligand:
                continue

            np_schemes = nonpoly_scheme[chain.chain_id]
            for seqres in chain.seqres:
                if seqres.res_id is None:
                    continue

                res = self.residues[seqres.res_id]
                scheme = Scheme(
                    asym_id=chain.chain_id,
                    entity_id=chain.entity_id,
                    mon_id=res.chem_comp.id,
                    seq_id=seqres.seq_id,
                    pdb_seq_num=res.residue_id.seq_id,
                    pdb_ins_code=res.residue_id.ins_code or None,
                )
                if scheme not in br_schemes:
                    np_schemes.add(scheme)

        label_seqs: dict[ResidueId, int] = {
            seqres.res_id: seqres.seq_id
            for chain in self.chains.values()
            for seqres in chain.seqres
            if seqres.res_id is not None
        }

        mmcif_content = (
            f"data_{name}"
            + mmcif_write_block(
                "entity",
                ["id", "type", "pdbx_description"],
                [
                    (entity.id, entity.type, entity.pdbx_description)
                    for entity in self.entities.values()
                ],
            )
            + mmcif_write_block(
                "struct_asym",
                ["id", "entity_id"],
                [(cid, chain.entity_id) for cid, chain in self.chains.items()],
            )
            + mmcif_write_block(
                "atom_site",
                [
                    "id",
                    "type_symbol",
                    "label_atom_id",
                    "label_comp_id",
                    "label_asym_id",
                    "label_seq_id",
                    "auth_seq_id",
                    "pdbx_PDB_ins_code",
                    "Cartn_x",
                    "Cartn_y",
                    "Cartn_z",
                ],
                [
                    (
                        atom.atom_idx,
                        atom.type_symbol,
                        atom.atom_id,
                        atom.comp_id,
                        atom.chain_id,
                        label_seqs[atom.residue_id],
                        atom.residue_id.seq_id,
                        atom.residue_id.ins_code or ".",
                        f"{crd[0]:.3f}",
                        f"{crd[1]:.3f}",
                        f"{crd[2]:.3f}",
                    )
                    for atom, crd in zip(self.atoms, self.coords)
                ],
            )
            + mmcif_write_block(
                "chem_comp",
                ["id", "type", "mon_nstd_flag"],
                [
                    (
                        comp.id,
                        comp.type,
                        "y" if comp.mon_nstd_flag else ".",
                    )
                    for comp in self.chem_comps.values()
                ],
            )
            + mmcif_write_block(
                "chem_comp_atom",
                [
                    "comp_id",
                    "atom_id",
                    "type_symbol",
                    "pdbx_aromatic_flag",
                    "pdbx_stereo_config",
                ],
                [
                    (
                        comp_id,
                        atom.atom_id,
                        atom.type_symbol,
                        mmcif_bool(atom.pdbx_aromatic_flag),
                        atom.pdbx_stereo_config or "N",
                    )
                    for comp_id, comp in self.chem_comps.items()
                    for atom in comp.atoms
                ],
            )
            + mmcif_write_block(
                "chem_comp_bond",
                [
                    "comp_id",
                    "atom_id_1",
                    "atom_id_2",
                    "value_order",
                    "pdbx_aromatic_flag",
                    "pdbx_stereo_config",
                ],
                [
                    (
                        comp_id,
                        bond.atom_id_1,
                        bond.atom_id_2,
                        mmcif_bond_order(bond.value_order),
                        mmcif_bool(bond.pdbx_aromatic_flag),
                        bond.pdbx_stereo_config or "N",
                    )
                    for comp_id, comp in self.chem_comps.items()
                    for bond in comp.bonds
                ],
            )
            + mmcif_write_block(
                "struct_conn",
                [
                    "conn_type_id",
                    "pdbx_leaving_atom_flag",
                    "ptnr1_label_asym_id",
                    "ptnr1_label_comp_id",
                    "ptnr1_label_atom_id",
                    "ptnr1_label_seq_id",
                    "ptnr1_auth_seq_id",
                    "pdbx_ptnr1_PDB_ins_code",
                    "ptnr2_label_asym_id",
                    "ptnr2_label_comp_id",
                    "ptnr2_label_atom_id",
                    "ptnr2_label_seq_id",
                    "ptnr2_auth_seq_id",
                    "pdbx_ptnr2_PDB_ins_code",
                ],
                [
                    (
                        conn.conn_type,
                        {0: "none", 1: "one", 2: "both"}[
                            conn.leaving_atom_count
                        ],
                        (ptnr1 := self.atoms[conn.src_idx]).chain_id,
                        ptnr1.comp_id,
                        ptnr1.atom_id,
                        label_seqs[ptnr1.residue_id],
                        ptnr1.residue_id.seq_id,
                        ptnr1.residue_id.ins_code or ".",
                        (ptnr2 := self.atoms[conn.dst_idx]).chain_id,
                        ptnr2.comp_id,
                        ptnr2.atom_id,
                        label_seqs[ptnr2.residue_id],
                        ptnr2.residue_id.seq_id,
                        ptnr2.residue_id.ins_code or ".",
                    )
                    for conn in self.connections
                ],
            )
            + mmcif_write_block(
                "pdbx_entity_branch_link",
                [
                    "entity_id",
                    "entity_branch_list_num_1",
                    "comp_id_1",
                    "atom_id_1",
                    "leaving_atom_id_1",
                    "entity_branch_list_num_2",
                    "comp_id_2",
                    "atom_id_2",
                    "leaving_atom_id_2",
                    "value_order",
                ],
                [
                    (
                        entity_id,
                        branch.ptnr1.seq_id,
                        branch.ptnr1.comp_id,
                        branch.ptnr1.atom_id,
                        branch.ptnr1.leaving_atom_id,
                        branch.ptnr2.seq_id,
                        branch.ptnr2.comp_id,
                        branch.ptnr2.atom_id,
                        branch.ptnr2.leaving_atom_id,
                        mmcif_bond_order(branch.order),
                    )
                    for entity_id, branches in merged_branches.items()
                    for branch in branches
                ],
            )
        )

        if write_schemes:
            mmcif_content += (
                mmcif_write_block(
                    "pdbx_poly_seq_scheme",
                    [
                        "asym_id",
                        "entity_id",
                        "seq_id",
                        "mon_id",
                        "pdb_seq_num",
                        "pdb_strand_id",
                        "pdb_ins_code",
                    ],
                    [
                        (
                            cid,
                            chain.entity_id,
                            seqres.seq_id,
                            seqres.comp_id,
                            seqres.res_id.seq_id if seqres.res_id else "?",
                            cid,
                            (seqres.res_id.ins_code or ".")
                            if seqres.res_id
                            else "?",
                        )
                        for cid, chain in self.chains.items()
                        if chain.type.is_polymer
                        for seqres in chain.seqres
                    ],
                )
                + mmcif_write_block(
                    "pdbx_branch_scheme",
                    [
                        "asym_id",
                        "entity_id",
                        "num",
                        "mon_id",
                        "pdb_seq_num",
                        "pdb_ins_code",
                    ],
                    [
                        (
                            scheme.asym_id,
                            scheme.entity_id,
                            scheme.seq_id,
                            scheme.mon_id,
                            scheme.pdb_seq_num,
                            scheme.pdb_ins_code or ".",
                        )
                        for schemes in branch_scheme.values()
                        for scheme in sorted(schemes)
                        if scheme.pdb_seq_num is not None
                    ],
                )
                + mmcif_write_block(
                    "pdbx_nonpoly_scheme",
                    [
                        "asym_id",
                        "entity_id",
                        "mon_id",
                        "ndb_seq_num",
                        "pdb_seq_num",
                        "pdb_ins_code",
                    ],
                    [
                        (
                            scheme.asym_id,
                            scheme.entity_id,
                            scheme.mon_id,
                            scheme.seq_id,
                            scheme.pdb_seq_num,
                            scheme.pdb_ins_code or ".",
                        )
                        for schemes in nonpoly_scheme.values()
                        for scheme in sorted(schemes)
                        if scheme.pdb_seq_num is not None
                    ],
                )
            )

        return mmcif_content + "\n#\n"

    def to_mmcif_chain(self, name: str, cid: str, write_schemes: bool = True):
        chain = self.chains[cid]

        merged_branches = set(chain.branches)
        branch_scheme = {
            branch.ptnr1.to_scheme(chain.entity_id)
            for branch in chain.branches
        } | {
            branch.ptnr2.to_scheme(chain.entity_id)
            for branch in chain.branches
        }
        nonpoly_scheme: set[Scheme] = set()
        if chain.type == MolType.Ligand:
            for seqres in chain.seqres:
                if seqres.res_id is None:
                    continue

                res = self.residues[seqres.res_id]
                scheme = Scheme(
                    asym_id=cid,
                    entity_id=chain.entity_id,
                    mon_id=res.chem_comp.id,
                    seq_id=seqres.seq_id,
                    pdb_seq_num=res.residue_id.seq_id,
                    pdb_ins_code=res.residue_id.ins_code or None,
                )
                if scheme not in branch_scheme:
                    nonpoly_scheme.add(scheme)

        label_seqs = {
            seqres.res_id: seqres.seq_id
            for seqres in chain.seqres
            if seqres.res_id is not None
        }
        entity = self.entities[chain.entity_id]

        ccid_chain = {
            self.residues[rid].chem_comp.id for rid in chain.residue_ids
        }
        atoms_chain = set(chain.atom_idxs)

        mmcif_content = (
            f"data_{name}_{cid}"
            + mmcif_write_block(
                "entity",
                ["id", "type", "pdbx_description"],
                [(entity.id, entity.type, entity.pdbx_description)],
            )
            + mmcif_write_block(
                "struct_asym",
                ["id", "entity_id"],
                [(cid, chain.entity_id)],
            )
            + mmcif_write_block(
                "atom_site",
                [
                    "id",
                    "type_symbol",
                    "label_atom_id",
                    "label_comp_id",
                    "label_asym_id",
                    "label_seq_id",
                    "auth_seq_id",
                    "pdbx_PDB_ins_code",
                    "Cartn_x",
                    "Cartn_y",
                    "Cartn_z",
                ],
                [
                    (
                        atom.atom_idx,
                        atom.type_symbol,
                        atom.atom_id,
                        atom.comp_id,
                        atom.chain_id,
                        label_seqs[atom.residue_id],
                        atom.residue_id.seq_id,
                        atom.residue_id.ins_code or ".",
                        f"{crd[0]:.3f}",
                        f"{crd[1]:.3f}",
                        f"{crd[2]:.3f}",
                    )
                    for atom, crd in zip(
                        self.atoms_of_chain(chain),
                        self.coords[chain.atom_idxs],
                    )
                ],
            )
            + mmcif_write_block(
                "chem_comp",
                ["id", "type", "mon_nstd_flag"],
                [
                    (
                        comp.id,
                        comp.type,
                        "y" if comp.mon_nstd_flag else ".",
                    )
                    for comp in self.chem_comps.values()
                    if comp.id in ccid_chain
                ],
            )
            + mmcif_write_block(
                "chem_comp_atom",
                [
                    "comp_id",
                    "atom_id",
                    "type_symbol",
                    "pdbx_aromatic_flag",
                    "pdbx_stereo_config",
                ],
                [
                    (
                        comp_id,
                        atom.atom_id,
                        atom.type_symbol,
                        mmcif_bool(atom.pdbx_aromatic_flag),
                        atom.pdbx_stereo_config or "N",
                    )
                    for comp_id, comp in self.chem_comps.items()
                    if comp.id in ccid_chain
                    for atom in comp.atoms
                ],
            )
            + mmcif_write_block(
                "chem_comp_bond",
                [
                    "comp_id",
                    "atom_id_1",
                    "atom_id_2",
                    "value_order",
                    "pdbx_aromatic_flag",
                    "pdbx_stereo_config",
                ],
                [
                    (
                        comp_id,
                        bond.atom_id_1,
                        bond.atom_id_2,
                        mmcif_bond_order(bond.value_order),
                        mmcif_bool(bond.pdbx_aromatic_flag),
                        bond.pdbx_stereo_config or "N",
                    )
                    for comp_id, comp in self.chem_comps.items()
                    if comp.id in ccid_chain
                    for bond in comp.bonds
                ],
            )
            + mmcif_write_block(
                "struct_conn",
                [
                    "conn_type_id",
                    "pdbx_leaving_atom_flag",
                    "ptnr1_label_asym_id",
                    "ptnr1_label_comp_id",
                    "ptnr1_label_atom_id",
                    "ptnr1_label_seq_id",
                    "ptnr1_auth_seq_id",
                    "pdbx_ptnr1_PDB_ins_code",
                    "ptnr2_label_asym_id",
                    "ptnr2_label_comp_id",
                    "ptnr2_label_atom_id",
                    "ptnr2_label_seq_id",
                    "ptnr2_auth_seq_id",
                    "pdbx_ptnr2_PDB_ins_code",
                ],
                [
                    (
                        conn.conn_type,
                        {0: "none", 1: "one", 2: "both"}[
                            conn.leaving_atom_count
                        ],
                        (ptnr1 := self.atoms[conn.src_idx]).chain_id,
                        ptnr1.comp_id,
                        ptnr1.atom_id,
                        label_seqs[ptnr1.residue_id],
                        ptnr1.residue_id.seq_id,
                        ptnr1.residue_id.ins_code or ".",
                        (ptnr2 := self.atoms[conn.dst_idx]).chain_id,
                        ptnr2.comp_id,
                        ptnr2.atom_id,
                        label_seqs[ptnr2.residue_id],
                        ptnr2.residue_id.seq_id,
                        ptnr2.residue_id.ins_code or ".",
                    )
                    for conn in self.connections
                    if conn.src_idx in atoms_chain
                    and conn.dst_idx in atoms_chain
                ],
            )
            + mmcif_write_block(
                "pdbx_entity_branch_link",
                [
                    "entity_id",
                    "entity_branch_list_num_1",
                    "comp_id_1",
                    "atom_id_1",
                    "leaving_atom_id_1",
                    "entity_branch_list_num_2",
                    "comp_id_2",
                    "atom_id_2",
                    "leaving_atom_id_2",
                    "value_order",
                ],
                [
                    (
                        chain.entity_id,
                        branch.ptnr1.seq_id,
                        branch.ptnr1.comp_id,
                        branch.ptnr1.atom_id,
                        branch.ptnr1.leaving_atom_id,
                        branch.ptnr2.seq_id,
                        branch.ptnr2.comp_id,
                        branch.ptnr2.atom_id,
                        branch.ptnr2.leaving_atom_id,
                        mmcif_bond_order(branch.order),
                    )
                    for branch in merged_branches
                ],
            )
        )

        if write_schemes:
            if chain.type.is_polymer:
                mmcif_content += mmcif_write_block(
                    "pdbx_poly_seq_scheme",
                    [
                        "asym_id",
                        "entity_id",
                        "seq_id",
                        "mon_id",
                        "pdb_seq_num",
                        "pdb_strand_id",
                        "pdb_ins_code",
                    ],
                    [
                        (
                            cid,
                            chain.entity_id,
                            seqres.seq_id,
                            seqres.comp_id,
                            seqres.res_id.seq_id if seqres.res_id else "?",
                            cid,
                            (seqres.res_id.ins_code or ".")
                            if seqres.res_id
                            else "?",
                        )
                        for seqres in chain.seqres
                    ],
                )

            mmcif_content += mmcif_write_block(
                "pdbx_branch_scheme",
                [
                    "asym_id",
                    "entity_id",
                    "num",
                    "mon_id",
                    "pdb_seq_num",
                    "pdb_ins_code",
                ],
                [
                    (
                        scheme.asym_id,
                        scheme.entity_id,
                        scheme.seq_id,
                        scheme.mon_id,
                        scheme.pdb_seq_num,
                        scheme.pdb_ins_code or ".",
                    )
                    for scheme in sorted(branch_scheme)
                    if scheme.pdb_seq_num is not None
                ],
            ) + mmcif_write_block(
                "pdbx_nonpoly_scheme",
                [
                    "asym_id",
                    "entity_id",
                    "mon_id",
                    "ndb_seq_num",
                    "pdb_seq_num",
                    "pdb_ins_code",
                ],
                [
                    (
                        scheme.asym_id,
                        scheme.entity_id,
                        scheme.mon_id,
                        scheme.seq_id,
                        scheme.pdb_seq_num,
                        scheme.pdb_ins_code or ".",
                    )
                    for scheme in sorted(nonpoly_scheme)
                    if scheme.pdb_seq_num is not None
                ],
            )

        return mmcif_content + "\n#\n"

    def to_pdb(self) -> str:
        if not np.all(np.isfinite(self.coords)) or np.any(
            (-1e3 >= self.coords) | (self.coords >= 1e4)
        ):
            raise ValueError("Coordinate value out of range")

        chain_map: dict[str, str] = {}
        chain_ids = list(self.chains.keys())
        chain_ids_avail = dict.fromkeys(
            ascii_uppercase + ascii_lowercase + digits,
            True,
        )
        if len(chain_ids_avail) < len(chain_ids):
            raise ValueError(
                (
                    "Cannot assign unique chain IDs for PDB output "
                    f"for {len(chain_ids)} chains"
                )
            )

        cid_need_remap: list[str] = []
        for cid in chain_ids:
            pdb_cid = cid[:1]
            if chain_ids_avail.pop(pdb_cid, False):
                chain_map[cid] = pdb_cid
            else:
                cid_need_remap.append(cid)

        for cid, pdb_cid in zip(cid_need_remap, chain_ids_avail):
            chain_map[cid] = pdb_cid

        def _resolve_atom_names(atom_id: str, element: str) -> str:
            if (
                len(element) == 1
                and len(atom_id) < 4
                and atom_id.upper().startswith(element.upper())
            ):
                return f" {atom_id:<3}"
            return atom_id.ljust(4)[:4]

        residue_atom_names: dict[ResidueId, dict[str, int]] = defaultdict(dict)

        atoms = sorted(self.atoms, key=lambda a: (a.residue_id, a.atom_idx))
        for atom in atoms:
            res_atom_names = residue_atom_names[atom.residue_id]
            atom_name = _resolve_atom_names(atom.atom_id, atom.type_symbol)
            idx = res_atom_names.setdefault(atom_name, atom.atom_idx)
            if idx != atom.atom_idx:
                raise ValueError(
                    (
                        f"Duplicate atom name {atom_name!r} "
                        f"in residue {atom.residue_id}"
                    )
                )

        atom_names = {
            i: name
            for atom_names in residue_atom_names.values()
            for name, i in atom_names.items()
        }

        def _record_name_for(
            residue_chem_comp: ChemComp,
            atom_chain_type: MolType,
        ) -> str:
            if atom_chain_type in (MolType.RNA, MolType.DNA):
                return "ATOM  "

            if residue_mol_type(residue_chem_comp) != MolType.Protein:
                return "HETATM"

            comp_id = residue_chem_comp.id
            is_standard = comp_id in aa_restype_3to1
            if not is_standard:
                return "HETATM"

            if residue_chem_comp.mon_nstd_flag:
                return "HETATM"

            return "ATOM  "

        chain_atoms: dict[str, list[_PdbAtom]] = defaultdict(list)
        for atom in atoms:
            record = _record_name_for(
                self.residues[atom.residue_id].chem_comp,
                self.chains[atom.chain_id].type,
            )

            ch = chain_map[atom.chain_id]
            rid = ResidueId(
                ch,
                atom.residue_id.seq_id,
                atom.residue_id.ins_code,
            )
            res_name = atom.comp_id[:3]
            atom_name = atom_names[atom.atom_idx]

            chain_atoms[ch].append(
                _PdbAtom(
                    record=record,
                    res_id=rid,
                    name=atom_name,
                    res_name=res_name,
                    coords=self.coords[atom.atom_idx],
                    element=atom.type_symbol,
                )
            )

        lines: list[str] = []
        lines.append("REMARK 950 CHAIN ID MAP (label_asym_id -> PDB chain)")
        for cid, one in chain_map.items():
            lines.append(f"REMARK 950   {one} = {cid[:66]}")

        serial = 1
        for ch_atoms in chain_atoms.values():
            for ca in ch_atoms:
                lines.append(ca.to_pdb_line(serial))
                serial += 1

            last_atom = ch_atoms[-1]
            res_name = last_atom.res_name
            res_id = last_atom.res_id
            lines.append(
                (
                    #                  1
                    #                  234567
                    f"TER   {serial:5d}      {res_name:>3} "
                    f"{res_id.chain_id}{res_id.seq_id:4d}{res_id.ins_code:1}"
                )
            )
            serial += 1

            if serial > 99999 or serial < -9999:
                raise ValueError("PDB serial number out of range")

        lines.append("END\n")

        return "\n".join(lines)

    def to_fasta(self, name: str) -> str:
        fastas: list[str] = []
        for chain_id, chain in self.chains.items():
            if not chain.type.is_polymer:
                continue

            seq_list = []
            for seqres in chain.seqres:
                comp_id = seqres.comp_id

                if chain.type == MolType.Protein:
                    comp_id = modres.get(comp_id, comp_id)
                    seq_list.append(aa_restype_3to1.get(comp_id, "X"))
                elif chain.type == MolType.DNA:
                    seq_list.append(dna_restype_3to1.get(comp_id, "N"))
                elif chain.type == MolType.RNA:
                    seq_list.append(rna_restype_3to1.get(comp_id, "N"))

            seq = "".join(seq_list)
            fastas.append(f">{name}_{chain_id}\n{seq}\n")
        return "".join(fastas)


def _resolve_residue_altloc_consistent(altlocs: dict[str, list[AtomSite]]):
    best_atom = max(
        (atom for atoms in altlocs.values() for atom in atoms),
        key=lambda a: a.occupancy,
    )
    comp_id = best_atom.label_comp_id

    for atoms in altlocs.values():
        consistent = [atom for atom in atoms if atom.label_comp_id == comp_id]
        if not consistent:
            continue

        yield max(consistent, key=lambda a: a.occupancy)


def _select_altlocs(atom_sites: list[AtomSite]):
    altlocs: dict[tuple[str, int, str | None], dict[str, list[AtomSite]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for atom in atom_sites:
        altlocs[atom.label_asym_id, atom.auth_seq_id, atom.pdbx_PDB_ins_code][
            atom.label_atom_id
        ].append(atom)

    return [
        atom
        for residue_altlocs in altlocs.values()
        for atom in _resolve_residue_altloc_consistent(residue_altlocs)
    ]


def residue_mol_type(chem_comp: ChemComp) -> MolType:
    chem_type = chem_comp.type.lower()

    if "peptide" in chem_type:
        return MolType.Protein

    if "rna" in chem_type:
        return MolType.RNA
    if "dna" in chem_type:
        return MolType.DNA

    return MolType.Ligand


def polymer_mol_type(
    seq_infos: list[Scheme],
    chem_comps: dict[str, ChemComp],
) -> MolType:
    molecule_types = {
        residue_mol_type(cc)
        for seq_info in seq_infos
        if (cc := chem_comps.get(seq_info.mon_id)) is not None
    }

    if MolType.Protein in molecule_types:
        return MolType.Protein

    if MolType.RNA in molecule_types:
        return MolType.RNA
    if MolType.DNA in molecule_types:
        return MolType.DNA

    raise ValueError(f"Invalid polymer chain type: {molecule_types}")


def mmcif_chain_types(
    metadata: Mmcif, ccd: dict[str, ChemComp]
) -> dict[str, MolType]:
    chain_types: dict[str, MolType] = {}

    for chain_id, seq_infos in metadata.pdbx_poly_seq_scheme.items():
        chain_types[chain_id] = polymer_mol_type(seq_infos, ccd)

    for chain_id in metadata.pdbx_branch_scheme:
        chain_types[chain_id] = MolType.Ligand

    for chain_id in metadata.pdbx_nonpoly_scheme:
        chain_types[chain_id] = MolType.Ligand

    return chain_types


def _map_struct_conn(
    conn: StructConn,
    chains: dict[str, Chain],
    residues: dict[ResidueId, Residue],
    atoms: list[AssemblyAtom],
) -> AssemblyConnection | None:
    idxs: list[int] = []

    for ptnr in (conn.ptnr1, conn.ptnr2):
        if ptnr.label_asym_id not in chains:
            return None

        residue = residues.get(
            ResidueId(
                ptnr.label_asym_id,
                ptnr.auth_seq_id,
                ptnr.pdbx_PDB_ins_code or "",
            ),
            None,
        )
        if residue is None or residue.chem_comp.id != ptnr.label_comp_id:
            return None

        for atom_idx in residue.atom_idxs:
            if atoms[atom_idx].atom_id == ptnr.label_atom_id:
                idxs.append(atom_idx)
                break
        else:
            return None

    return AssemblyConnection(
        src_idx=idxs[0],
        dst_idx=idxs[1],
        conn_type=conn.conn_type_id.lower(),
        leaving_atom_count=conn.pdbx_leaving_atom_flag,
    )


def _map_branches(
    br_schemes: list[Scheme],
    br_links: list[BranchLink],
    residues: dict[ResidueId, Residue],
):
    schemes = {scheme.seq_id: scheme for scheme in br_schemes}

    branches = []
    for link in br_links:
        ptnr1 = BranchPartner.from_branch_scheme(
            schemes[link.ptnr1.entity_branch_list_num],
            link.ptnr1,
        )
        ptnr2 = BranchPartner.from_branch_scheme(
            schemes[link.ptnr2.entity_branch_list_num],
            link.ptnr2,
        )
        if (
            residues[ptnr1.res_id].chem_comp.id != ptnr1.comp_id
            or residues[ptnr2.res_id].chem_comp.id != ptnr2.comp_id
        ):
            continue

        branches.append(Branch(ptnr1, ptnr2, link.value_order))

    return branches


def _select_het_seq_scheme(
    schemes: list[Scheme],
    residues: dict[ResidueId, Residue],
):
    seqres: list[SequenceToResidue] = []

    for scheme in schemes:
        s2r = SequenceToResidue.from_scheme(scheme)
        if (
            s2r.res_id is not None
            and (res := residues.get(s2r.res_id, None)) is not None
            and res.chem_comp.id != s2r.comp_id
        ):
            continue

        seqres.append(s2r)

    return seqres


def _model_assembly(
    metadata: Mmcif, atom_sites: list[AtomSite], ccd: dict[str, ChemComp]
) -> Assembly:
    atom_sites = _select_altlocs(atom_sites)

    atom_sites = [
        atom_site for atom_site in atom_sites if atom_site.label_comp_id in ccd
    ]

    coords = np.stack([atom_site.cartn for atom_site in atom_sites])

    atoms = [
        AssemblyAtom(
            atom_idx=i,
            residue_id=ResidueId(
                chain_id=atom_site.label_asym_id,
                seq_id=atom_site.auth_seq_id,
                ins_code=atom_site.pdbx_PDB_ins_code or "",
            ),
            type_symbol=atom_site.type_symbol,
            atom_id=atom_site.label_atom_id,
            comp_id=atom_site.label_comp_id,
        )
        for i, atom_site in enumerate(atom_sites)
    ]

    residues: dict[ResidueId, Residue] = {}
    chains: dict[str, Chain] = {}

    chain_types = mmcif_chain_types(metadata, ccd)

    for atom in atoms:
        chain = chains.setdefault(
            atom.chain_id,
            Chain(
                atom.chain_id,
                metadata.struct_asym[atom.chain_id],
                chain_types[atom.chain_id],
            ),
        )

        if (residue := residues.get(atom.residue_id, None)) is None:
            residue = residues[atom.residue_id] = Residue(
                atom.residue_id, ccd[atom.comp_id]
            )

            chain.residue_ids.append(residue.residue_id)

        chain.atom_idxs.append(atom.atom_idx)
        residue.atom_idxs.append(atom.atom_idx)

    for cid, chain in chains.items():
        chain.seqres = _select_het_seq_scheme(
            metadata.pdbx_poly_seq_scheme.get(cid, [])
            + metadata.pdbx_branch_scheme.get(cid, [])
            + metadata.pdbx_nonpoly_scheme.get(cid, []),
            residues,
        )
        chain.branches = _map_branches(
            metadata.pdbx_branch_scheme.get(cid, []),
            metadata.pdbx_entity_branch_link.get(chain.entity_id, []),
            residues,
        )
        chain.sync_mappings()

    connections = [
        ac
        for conn in metadata.struct_conn
        if (ac := _map_struct_conn(conn, chains, residues, atoms)) is not None
    ]

    return Assembly(
        coords,
        atoms,
        residues,
        chains,
        metadata.entity.copy(),
        connections,
    )


def _prepare_initial_assemblies(
    metadata: Mmcif, ccd: dict[str, ChemComp]
) -> list[Assembly]:
    atom_sites_model: dict[int, list[AtomSite]] = defaultdict(list)
    for atom_site in metadata.atom_site:
        if atom_site.type_symbol == "H":
            continue

        atom_sites_model[atom_site.pdbx_PDB_model_num].append(atom_site)

    return [
        _model_assembly(metadata, atom_sites, ccd)
        for atom_sites in atom_sites_model.values()
    ]


def _smallest_assembly_gens(bas: list[BioAssembly]):
    min_oli_cnt = min(bas, key=lambda ba: ba.oligomeric_count).oligomeric_count
    return [ba for ba in bas if ba.oligomeric_count == min_oli_cnt]


def _first_assembly_gens(bas: list[BioAssembly], assembly_id: int):
    return [ba for ba in bas if ba.id == assembly_id]


def _select_assembly_gens(
    bas: list[BioAssembly], assembly_id: int | None = None
):
    return (
        _first_assembly_gens(bas, assembly_id)
        if assembly_id is not None
        else _smallest_assembly_gens(bas)
    )


def expand_ops(operations: list[list[str]], sym_ops: dict[str, SymOp]):
    return [
        sum(
            (Transformation.from_symop(sym_ops[op]) for op in combi),
            Transformation.ident,
        )
        for combi in itertools.product(*operations)
    ]


def _apply_transforms(
    assem: Assembly,
    generators: list[tuple[list[str], list[Transformation]]],
) -> Assembly:
    assemblies = [
        assem.filter_chains(chain_ids).apply(xforms)
        for chain_ids, xforms in generators
    ]
    return Assembly.join(assemblies)


def mmcif_assemblies(
    data: Mmcif, ccd: dict[str, ChemComp], assembly_id: int | None = None
) -> list[Assembly]:
    assemblies = _prepare_initial_assemblies(data, ccd)
    raw_assemblies = _select_assembly_gens(
        data.pdbx_struct_assembly, assembly_id
    )

    operation_groups = [
        [
            (
                g.asym_id_list,
                expand_ops(g.operations, data.pdbx_struct_oper_list),
            )
            for g in ra.assembly_gens
        ]
        for ra in raw_assemblies
    ]

    return [
        _apply_transforms(initial_assembly, ops)
        for initial_assembly in assemblies
        for ops in operation_groups
    ]
