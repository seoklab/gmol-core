import datetime as dt
import enum
import gzip
import logging
from collections.abc import Set
from dataclasses import InitVar, field
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, TypeAdapter
from rdkit import Chem
from scipy.spatial import distance as D

from gmol.base import const
from .assembly import (
    Assembly,
    Chain,
    MolType,
    Residue,
    ResidueId,
    mmcif_assemblies,
)
from .filter import filter_mmcif
from .parse import (
    ChemComp,
    ChemCompAtom,
    ChemCompBond,
    load_mmcif_single,
)
from .smiles import (
    MmcifRefLigand,
    RefLigandInput,
    input_from_reference,
    reference_from_mmcif,
    unique_atom_id,
)

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PolymerConstants:
    restype_3_to_1: dict[str, str]
    restype_order_with_x: dict[str, int]

    backbone: set[str]
    modres_backbone_3: str

    bb_src_atom: str
    bb_src_leaving: list[str]
    bb_dst_atom: str
    bb_dst_leaving: list[str]

    max_res_atoms: int

    residue_atom_idxs: dict[str, dict[str, int]] = field(init=False)
    restype_modres_backbone: int = field(init=False)
    restype_x: int = field(init=False)

    _modres_backbone_1: InitVar[str]
    _res_to_atom: InitVar[dict[str, list[str]]]

    def __post_init__(
        self,
        _modres_backbone_1: str,
        _res_to_atom: dict[str, list[str]],
    ):
        # OXT etc. must also be considered as backbone atoms
        self.backbone.update(self.bb_src_leaving)
        self.backbone.update(self.bb_dst_leaving)

        self.residue_atom_idxs = {
            res: {
                atom_name: i
                for i, atom_name in enumerate(atoms)
                if atom_name != ""
            }
            for res, atoms in _res_to_atom.items()
        }
        self.restype_modres_backbone = self.restype_order_with_x[
            _modres_backbone_1
        ]
        self.restype_x = self.restype_order_with_x["X"]


chain_type_to_constants = {
    MolType.Protein: PolymerConstants(
        restype_3_to_1={**const.aa_restype_3to1, "MSE": "M"},
        restype_order_with_x=const.aa_restype_order_with_x,
        backbone={"N", "CA", "C", "O"},
        _modres_backbone_1="G",
        modres_backbone_3="GLY",
        bb_src_atom="C",
        bb_src_leaving=["OXT", "HXT"],
        bb_dst_atom="N",
        bb_dst_leaving=["H"],
        max_res_atoms=14,
        _res_to_atom=const.aa_restype_to_atom14,
    ),
    MolType.RNA: PolymerConstants(
        restype_3_to_1=const.rna_restype_3to1,
        restype_order_with_x=const.rna_restype_order_with_x,
        backbone=set(const.rna_backbone),
        _modres_backbone_1="X",
        modres_backbone_3="N",
        bb_src_atom="O3'",
        bb_src_leaving=["HO3'"],
        bb_dst_atom="P",
        bb_dst_leaving=["OP3", "HOP3"],
        max_res_atoms=24,
        _res_to_atom=const.rna_restype_to_atom24,
    ),
    MolType.DNA: PolymerConstants(
        restype_3_to_1=const.dna_restype_3to1,
        restype_order_with_x=const.dna_restype_order_with_x,
        backbone=set(const.dna_backbone),
        _modres_backbone_1="X",
        modres_backbone_3="DN",
        bb_src_atom="O3'",
        bb_src_leaving=["HO3'"],
        bb_dst_atom="P",
        bb_dst_leaving=["OP3", "HOP3"],
        max_res_atoms=24,
        _res_to_atom=const.dna_restype_to_atom24,
    ),
}


class BondOrder(enum.IntEnum):
    other = 0
    single = 1
    double = 2
    triple = 3
    quadruple = 4
    aromatic = 5

    @classmethod
    def from_rdkit_bond_type(cls, bond_type: Chem.BondType):
        mapping = {
            Chem.BondType.SINGLE: cls.single,
            Chem.BondType.DOUBLE: cls.double,
            Chem.BondType.TRIPLE: cls.triple,
            Chem.BondType.QUADRUPLE: cls.quadruple,
            Chem.BondType.AROMATIC: cls.aromatic,
        }
        return mapping.get(bond_type, cls.other)

    def to_rdkit_bond_type(self):
        mapping = {
            BondOrder.single: Chem.BondType.SINGLE,
            BondOrder.double: Chem.BondType.DOUBLE,
            BondOrder.triple: Chem.BondType.TRIPLE,
            BondOrder.quadruple: Chem.BondType.QUADRUPLE,
            BondOrder.aromatic: Chem.BondType.AROMATIC,
        }
        return mapping.get(self, Chem.BondType.UNSPECIFIED)

    @classmethod
    def from_mmcif_value_order(cls, value_order: int):
        mapping = {
            1: cls.single,
            2: cls.double,
            3: cls.triple,
            4: cls.quadruple,
        }
        return mapping.get(value_order, cls.other)


class BondClass(enum.IntEnum):
    unknown = 0
    covalent = 1
    disulfide = 2
    hydrogen = 3
    metal_coordinate = 4

    @classmethod
    def from_mmcif_conn_type(cls, conn_type: str):
        mapping = {
            "covale": cls.covalent,
            "disulf": cls.disulfide,
            "hydrog": cls.hydrogen,
            "metalc": cls.metal_coordinate,
        }
        return mapping.get(conn_type.lower(), cls.unknown)


@dataclass
class PolymerChain:
    mol_type: MolType

    entity_id: str
    chain_id: str

    # [N_res]
    # When set to X, modified residue (as a whole) is extracted as a ligand;
    # output corresponding to the residue must be ignored
    restype: NDArray[np.int8]
    # Raw chem_comp id for MSA generation; for modified residues
    chem_comp_raw: NDArray[np.str_]

    # [N_res, N_atom_type, 3]
    atom_coords: NDArray[np.float64]

    # Below is for reference, not used in input
    # [N_res]
    residue_ids: list[ResidueId | None]

    __pydantic_config__ = ConfigDict(arbitrary_types_allowed=True)

    def __len__(self):
        return len(self.restype)


@dataclass
class NonPolymerLigand:
    entity_id: str
    chain_id: str

    smiles: str

    # [N_atom]
    atom_ids: NDArray[np.str_]
    # [N_atom, 3]
    atom_coords: NDArray[np.float64]

    __pydantic_config__ = ConfigDict(arbitrary_types_allowed=True)


@dataclass(order=True, frozen=True)
class ExtraBondPartner:
    chain_id: str
    res_idx: int
    atom_id: str

    leaving_atoms: list[str] = field(default_factory=list, compare=False)


@dataclass
class ExtraBond:
    src: ExtraBondPartner
    dst: ExtraBondPartner

    bond_order: BondOrder
    bond_class: BondClass
    # E:-1. N/Undefined:0. Z:1  # Create a EZ stereo dataclass?
    bond_stereo_EZ: int = 0

    def __post_init__(self):
        if (
            self.bond_class != BondClass.covalent
            or self.bond_order != BondOrder.double
        ):
            assert self.bond_stereo_EZ == 0, (
                "No EZ Configuration except for double covalent bonds."
            )

    @staticmethod
    def stereo_from_mmcif(pdbx_stereo_config: str | None):
        return {"E": -1, "N": 0, None: 0, "Z": 1}.get(pdbx_stereo_config, 0)


@dataclass(frozen=True)
class _ModresFrom:
    chain: str
    res_idx: int
    atom_id: str


@dataclass
class _ModresTo:
    mod_chain: str
    res_idx: int


@dataclass
class Input:
    polymers: list[PolymerChain]
    ligands: list[NonPolymerLigand]
    extra_bonds: list[ExtraBond]

    release_date: dt.date
    resolution: float
    is_distillation: bool


def build_atom_coords_from_ref(
    assembly: Assembly,
    residues: list[Residue],
    ref_ligand: RefLigandInput,
) -> NDArray[np.float64]:
    coords = np.full((len(ref_ligand.atom_ids), 3), np.nan, dtype=np.float64)

    if len(residues) == 1:

        def resolve_atom_id(residue_id: ResidueId, atom_id: str):
            return atom_id
    else:

        def resolve_atom_id(residue_id: ResidueId, atom_id: str):
            return unique_atom_id(residue_id, atom_id)

    aid_to_index = {aid: i for i, aid in enumerate(ref_ligand.atom_ids)}
    for residue in residues:
        for atom in assembly.atoms_of_residue(residue):
            atom_id = resolve_atom_id(residue.residue_id, atom.atom_id)
            if atom_id not in aid_to_index:
                continue

            coords[aid_to_index[atom_id]] = assembly.coords[atom.atom_idx]

    return coords


def _update_polymer_residue_coords(
    coords: NDArray[np.float64],
    assembly: Assembly,
    residue: Residue,
    polymer_consts: PolymerConstants,
    well_known_atoms: Set[str],
    comp_id: str | None = None,
):
    comp_id = comp_id or residue.chem_comp.id
    if comp_id == "MSE":
        # alias SE to SD
        atom_idxs = {
            ("SE" if aid == "SD" else aid): idx
            for aid, idx in polymer_consts.residue_atom_idxs["MET"].items()
        }
    else:
        atom_idxs = polymer_consts.residue_atom_idxs[comp_id]

    for atom in assembly.atoms_of_residue(residue):
        try:
            coords[atom_idxs[atom.atom_id]] = assembly.coords[atom.atom_idx]
        except KeyError:
            if atom.atom_id not in well_known_atoms:
                logger.warning(
                    "Unknown atom %s in compound %s",
                    atom.atom_id,
                    comp_id,
                )

    if comp_id != "ARG":
        return

    # From AlphaFold2: swap NH1 and NH2 in ARG if NH2 is closer to CD than NH1
    i = atom_idxs["NH1"]
    j = atom_idxs["NH2"]
    cd_nh1_nh2 = coords[[atom_idxs["CD"], i, j]]
    if np.isnan(cd_nh1_nh2).any():
        return

    dsq_nh1 = D.sqeuclidean(cd_nh1_nh2[0], cd_nh1_nh2[1])
    dsq_nh2 = D.sqeuclidean(cd_nh1_nh2[0], cd_nh1_nh2[2])
    if dsq_nh1 > dsq_nh2:
        coords[[j, i]] = cd_nh1_nh2[1:]


def _add_implicit_bb_bonds(
    polymer: PolymerChain,
    polymer_consts: PolymerConstants,
    mod_chain: str,
    res_idx: int,
):
    bonds: list[ExtraBond] = []
    if res_idx > 0:
        bonds.append(
            ExtraBond(
                src=ExtraBondPartner(
                    chain_id=polymer.chain_id,
                    res_idx=res_idx - 1,
                    atom_id=polymer_consts.bb_src_atom,
                    leaving_atoms=polymer_consts.bb_src_leaving,
                ),
                dst=ExtraBondPartner(
                    chain_id=mod_chain,
                    res_idx=0,
                    atom_id=polymer_consts.bb_dst_atom,
                    leaving_atoms=polymer_consts.bb_dst_leaving,
                ),
                bond_order=BondOrder.single,
                bond_class=BondClass.covalent,
            )
        )
    if res_idx < len(polymer) - 1:
        bonds.append(
            ExtraBond(
                src=ExtraBondPartner(
                    chain_id=mod_chain,
                    res_idx=0,
                    atom_id=polymer_consts.bb_src_atom,
                    leaving_atoms=polymer_consts.bb_src_leaving,
                ),
                dst=ExtraBondPartner(
                    chain_id=polymer.chain_id,
                    res_idx=res_idx + 1,
                    atom_id=polymer_consts.bb_dst_atom,
                    leaving_atoms=polymer_consts.bb_dst_leaving,
                ),
                bond_order=BondOrder.single,
                bond_class=BondClass.covalent,
            )
        )

    return bonds


def _modres_as_ligand(
    assembly: Assembly,
    polymer: PolymerChain,
    polymer_consts: PolymerConstants,
    mod_entity: str,
    mod_chain: str,
    resid: ResidueId | None,
    res_idx: int,
    chem_comp: ChemComp,
):
    ref_mmcif = MmcifRefLigand(chem_comp.atoms, chem_comp.bonds)
    ref_ligand = input_from_reference(ref_mmcif)
    crd = build_atom_coords_from_ref(
        assembly,
        [] if resid is None else [assembly.residues[resid]],
        ref_ligand,
    )

    ligand = NonPolymerLigand(
        entity_id=mod_entity,
        chain_id=mod_chain,
        smiles=ref_ligand.smiles,
        atom_ids=np.array(ref_ligand.atom_ids, dtype=np.str_),
        atom_coords=crd,
    )
    bonds = _add_implicit_bb_bonds(polymer, polymer_consts, mod_chain, res_idx)

    atom_map = {
        _ModresFrom(
            chain=polymer.chain_id,
            res_idx=res_idx,
            atom_id=atom_id,
        ): _ModresTo(mod_chain=mod_chain, res_idx=0)
        for atom_id in ref_ligand.atom_ids
    }

    return ligand, bonds, atom_map


def _split_modified_residue(
    chain_consts: PolymerConstants,
    chem_comp: ChemComp,
    chain: str,
    res_idx: int,
    mod_chain_prefix: str,
) -> tuple[list[ChemComp], list[ExtraBond]]:
    graph = nx.Graph()
    graph.add_nodes_from([(a.atom_id, {"data": a}) for a in chem_comp.atoms])
    graph.add_edges_from(
        [(b.atom_id_1, b.atom_id_2, {"data": b}) for b in chem_comp.bonds]
    )

    bb_all = chain_consts.backbone.copy()
    for aid in chain_consts.backbone:
        for n in graph[aid]:
            atom: ChemCompAtom = graph.nodes[n]["data"]
            if atom.type_symbol == "H":
                bb_all.add(n)

    only_sc = graph.copy(as_view=False)
    only_sc.remove_nodes_from(bb_all)

    fragments: list[ChemComp] = []
    fragment_idxs: dict[str, int] = {}
    atom_ids: set[str]
    for i, atom_ids in enumerate(nx.connected_components(only_sc)):
        for atom_id in atom_ids:
            fragment_idxs[atom_id] = i

        frag_graph: nx.Graph = nx.induced_subgraph(only_sc, atom_ids)
        comp_atoms = [atom for _, atom in frag_graph.nodes(data="data")]
        comp_bonds = [bond for *_, bond in frag_graph.edges(data="data")]
        frag_cc = ChemComp(
            id=f"{chem_comp.id}-FRAG{i}",
            name=f"{chem_comp.name}-FRAG{i}",
            type=f"{chem_comp.type}, fragment {i}",
            formula=None,
            formula_weight=None,
            atoms=comp_atoms,
            bonds=comp_bonds,
        )
        fragments.append(frag_cc)

    inter_bonds: list[ExtraBond] = []
    hydrogen_counts = [0] * len(fragments)
    bond: ChemCompBond
    for src, dst, bond in nx.edge_boundary(graph, bb_all, data="data"):
        # TODO: handle value_order
        assert bond.value_order == 1

        fi = fragment_idxs[dst]

        inter_bonds.append(
            ExtraBond(
                src=ExtraBondPartner(
                    chain_id=chain,
                    res_idx=res_idx,
                    atom_id=src,
                ),
                dst=ExtraBondPartner(
                    chain_id=f"{mod_chain_prefix}-FRAG{fi}",
                    res_idx=0,
                    atom_id=dst,
                ),
                bond_order=BondOrder.from_mmcif_value_order(bond.value_order),
                bond_class=BondClass.covalent,
                bond_stereo_EZ=ExtraBond.stereo_from_mmcif(
                    bond.pdbx_stereo_config
                ),
            )
        )

        hname = f"{dst}-DUMMYH-{hydrogen_counts[fi]}"
        hydrogen_counts[fi] += 1

        fragment = fragments[fi]
        fragment.atoms.append(ChemCompAtom(atom_id=hname, type_symbol="H"))
        fragment.bonds.append(
            ChemCompBond(atom_id_1=dst, atom_id_2=hname, value_order=1)
        )

    return fragments, inter_bonds


def _build_sc_ligand(
    assembly: Assembly,
    resid: ResidueId | None,
    sc_frag: ChemComp,
    mod_entity: str,
    mod_chain: str,
) -> NonPolymerLigand:
    ref_mmcif = MmcifRefLigand(sc_frag.atoms, sc_frag.bonds)
    ref_ligand = input_from_reference(ref_mmcif)
    crd = build_atom_coords_from_ref(
        assembly,
        [] if resid is None else [assembly.residues[resid]],
        ref_ligand,
    )

    ligand = NonPolymerLigand(
        entity_id=mod_entity,
        chain_id=mod_chain,
        smiles=ref_ligand.smiles,
        atom_ids=np.array(ref_ligand.atom_ids, dtype=np.str_),
        atom_coords=crd,
    )
    return ligand


def _modres_split_bb_sc(
    assembly: Assembly,
    chain_consts: PolymerConstants,
    chain_id: str,
    resid: ResidueId | None,
    res_idx: int,
    chem_comp: ChemComp,
):
    mod_entity_prefix = f"X-MOD:{chem_comp.id}"
    mod_chain_prefix = f"{chain_id}-MOD{res_idx}"

    sidechain_frags, inter_bonds = _split_modified_residue(
        chain_consts,
        chem_comp,
        chain_id,
        res_idx,
        mod_chain_prefix,
    )
    sidechain_ligands = [
        _build_sc_ligand(
            assembly,
            resid,
            frag,
            f"{mod_entity_prefix}-FRAG{i}",
            f"{mod_chain_prefix}-FRAG{i}",
        )
        for i, frag in enumerate(sidechain_frags)
    ]

    atom_map = {
        _ModresFrom(
            chain=chain_id,
            res_idx=res_idx,
            atom_id=atom_id,
        ): _ModresTo(mod_chain=f"{mod_chain_prefix}-FRAG{i}", res_idx=0)
        for i, lig in enumerate(sidechain_ligands)
        for atom_id in lig.atom_ids
    }

    return sidechain_ligands, inter_bonds, atom_map


def process_polymer_chain(
    assembly: Assembly,
    chain: Chain,
    ccd: dict[str, ChemComp],
    split_modified: bool = False,
    well_known_atoms: Set[str] = frozenset({"OXT", "HXT"}),
):
    polymer_consts = chain_type_to_constants[chain.type]

    polymer = PolymerChain(
        mol_type=chain.type,
        entity_id=str(chain.entity_id),
        chain_id=chain.chain_id,
        restype=np.array(
            [
                polymer_consts.restype_order_with_x[
                    polymer_consts.restype_3_to_1.get(seq.comp_id, "X")
                ]
                for seq in chain.seqres
            ]
        ),
        chem_comp_raw=np.array(
            [seq.comp_id for seq in chain.seqres], dtype=np.str_
        ),
        atom_coords=np.full(
            (len(chain.seqres), polymer_consts.max_res_atoms, 3),
            np.nan,
            dtype=np.float64,
        ),
        residue_ids=[seq.res_id for seq in chain.seqres],
    )

    non_polymers: list[NonPolymerLigand] = []
    extra_bonds: list[ExtraBond] = []
    modres_atom_map: dict[_ModresFrom, _ModresTo] = {}

    restype: int
    for i, (seq, restype) in enumerate(zip(chain.seqres, polymer.restype)):
        # Standard residues
        if restype != polymer_consts.restype_x:
            if seq.res_id is not None:
                _update_polymer_residue_coords(
                    polymer.atom_coords[i],
                    assembly,
                    assembly.residues[seq.res_id],
                    polymer_consts,
                    well_known_atoms,
                )
            continue

        mod_entity = f"X-MOD:{seq.comp_id}"
        mod_chain = f"{chain.chain_id}-MOD{i}"

        # whole modified residue as ligand (no polymer)
        if not split_modified:
            lig, bonds, atom_map = _modres_as_ligand(
                assembly,
                polymer,
                polymer_consts,
                mod_entity,
                mod_chain,
                seq.res_id,
                i,
                ccd[seq.comp_id],
            )

            non_polymers.append(lig)
            extra_bonds.extend(bonds)
            modres_atom_map.update(atom_map)
            continue

        # split modified residue into backbone and sidechain "covalent ligands"
        polymer.restype[i] = polymer_consts.restype_modres_backbone

        if seq.res_id is not None:
            _update_polymer_residue_coords(
                polymer.atom_coords[i],
                assembly,
                assembly.residues[seq.res_id],
                polymer_consts,
                well_known_atoms,
                comp_id=polymer_consts.modres_backbone_3,
            )

        sidechain_ligands, inter_bonds_local, atom_map = _modres_split_bb_sc(
            assembly,
            polymer_consts,
            chain.chain_id,
            seq.res_id,
            i,
            ccd[seq.comp_id],
        )
        non_polymers.extend(sidechain_ligands)
        extra_bonds.extend(inter_bonds_local)
        modres_atom_map.update(atom_map)

    return polymer, non_polymers, extra_bonds, modres_atom_map


def process_ligand_chain(assembly: Assembly, chain: Chain):
    # TODO: handle ligand chain with multiple ligands
    ccs = [
        (residue.residue_id, residue.chem_comp)
        for residue in assembly.residues_of_chain(chain)
    ]
    lig = input_from_reference(reference_from_mmcif(ccs, chain.branches))
    residues = [assembly.residues[rid] for rid, _ in ccs]
    coords = build_atom_coords_from_ref(assembly, residues, lig)

    return NonPolymerLigand(
        chain_id=chain.chain_id,
        entity_id=str(chain.entity_id),
        smiles=lig.smiles,
        atom_ids=np.array(lig.atom_ids, dtype=np.str_),
        atom_coords=coords,
    )


def _resolve_partner(
    assembly: Assembly,
    modres_atom_map: dict[_ModresFrom, _ModresTo],
    atom_idx: int,
):
    atom = assembly.atoms[atom_idx]
    chain = assembly.chains[atom.chain_id]

    ptnr = ExtraBondPartner(
        chain_id=atom.chain_id,
        res_idx=chain.resseq[atom.residue_id],
        atom_id=atom.atom_id,
    )

    key = _ModresFrom(ptnr.chain_id, ptnr.res_idx, ptnr.atom_id)
    if (val := modres_atom_map.get(key, None)) is not None:
        ptnr = ExtraBondPartner(
            chain_id=val.mod_chain,
            res_idx=val.res_idx,
            atom_id=ptnr.atom_id,
        )
    return ptnr


def _is_implicit_backbone_bond(
    assembly: Assembly,
    src: ExtraBondPartner,
    dst: ExtraBondPartner,
):
    if src.chain_id != dst.chain_id:
        return False

    try:
        chain = assembly.chains[src.chain_id]
        polymer_consts = chain_type_to_constants[chain.type]
    except KeyError:
        return False

    src, dst = sorted([src, dst], key=lambda x: x.res_idx)
    return (
        src.res_idx + 1 == dst.res_idx
        and src.atom_id == polymer_consts.bb_src_atom
        and dst.atom_id == polymer_consts.bb_dst_atom
    )


def process_connections(
    assembly: Assembly,
    modres_atom_map: dict[_ModresFrom, _ModresTo],
) -> list[ExtraBond]:
    bonds: list[ExtraBond] = []

    for conn in assembly.connections:
        src = _resolve_partner(assembly, modres_atom_map, conn.src_idx)
        dst = _resolve_partner(assembly, modres_atom_map, conn.dst_idx)

        if _is_implicit_backbone_bond(assembly, src, dst):
            continue

        # TODO: handle leaving atoms
        bond = ExtraBond(
            src=src,
            dst=dst,
            bond_order=BondOrder.single,
            bond_class=BondClass.from_mmcif_conn_type(conn.conn_type),
        )
        bonds.append(bond)

    return bonds


def _bond_as_key(bond: ExtraBond):
    return tuple(
        sorted(
            [
                (bond.src.chain_id, bond.src.res_idx, bond.src.atom_id),
                (bond.dst.chain_id, bond.dst.res_idx, bond.dst.atom_id),
            ]
        )
    )


def build_input(
    assembly: Assembly,
    ccd: dict[str, ChemComp],
    split_modified: bool = False,
    well_known_atoms: Set[str] = frozenset({"OXT", "HXT"}),
) -> Input:
    # bonds for modified residues (backbone - sidechain)
    new_bonds_for_modres: list[ExtraBond] = []
    polymers_all: list[PolymerChain] = []
    non_polymers_all: list[NonPolymerLigand] = []
    modres_atom_map: dict[_ModresFrom, _ModresTo] = {}
    for chain in assembly.chains.values():
        if not chain.type.is_polymer:
            non_polymers_all.append(process_ligand_chain(assembly, chain))
            continue

        poly, nonpolys, modres_bonds, modres_map = process_polymer_chain(
            assembly,
            chain,
            ccd,
            split_modified,
            well_known_atoms,
        )
        polymers_all.append(poly)
        non_polymers_all.extend(nonpolys)
        new_bonds_for_modres.extend(modres_bonds)
        modres_atom_map.update(modres_map)

    explicit_bonds = process_connections(assembly, modres_atom_map)

    mmcif_bond_index = {_bond_as_key(bond): bond for bond in explicit_bonds}
    for bond in new_bonds_for_modres:
        if _bond_as_key(bond) in mmcif_bond_index:
            # See seoklab/gmol-base#9 for context.
            logger.debug("Duplicate bond: %s", bond)
        else:
            explicit_bonds.append(bond)

    return Input(
        polymers=polymers_all,
        ligands=non_polymers_all,
        extra_bonds=explicit_bonds,
        release_date=assembly.metadata.revision_date,
        resolution=assembly.metadata.resolution,
        is_distillation=False,
    )


def build_input_from_mmcif(
    mmcif_path: str,
    ccd_comp: dict[str, ChemComp],
) -> Input | None:
    metadata = load_mmcif_single(Path(mmcif_path))
    assemblies = mmcif_assemblies(metadata, ccd_comp)
    filtered_assembly = filter_mmcif(assemblies[0], ccd_comp)

    if filtered_assembly is not None:
        return build_input(filtered_assembly, ccd_comp, split_modified=False)

    return None


def load_ccd_dict_json(file_path: Path) -> dict[str, ChemComp]:
    with gzip.open(file_path) as f:
        ccd_comp = TypeAdapter(dict[str, ChemComp]).validate_json(f.read())
    return ccd_comp
