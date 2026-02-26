"""
cif_parse() and related subroutines are based on the BioPython library.
Here follows the full license text from BioPython:


Copyright (C) 2002, Thomas Hamelryck (thamelry@binf.ku.dk)

Copyright (c) 1999-2024, The Biopython Contributors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

# mypy: disallow-any-explicit=false

import datetime as dt
import logging
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from nuri.fmt import cif_ddl2_frame_as_dict, read_cif
from pydantic import (
    AliasChoices,
    AliasPath,
    BaseModel,
    Field,
    field_validator,
    model_validator,
)
from tqdm import tqdm

from gmol.base.types import LooseModel

__all__ = ["ChemComp", "Mmcif", "load_components", "load_mmcif_single"]

_logger = logging.getLogger(__name__)


class Entity(LooseModel):
    id: int
    type: str
    pdbx_description: str


class ChemCompAtom(LooseModel):
    atom_id: str
    type_symbol: str
    charge: float = 0.0
    pdbx_aromatic_flag: bool = False
    pdbx_leaving_atom_flag: bool = False
    pdbx_stereo_config: str | None = None

    @field_validator("type_symbol", mode="after")
    @staticmethod
    def _correct_case(v: str) -> str:
        return v.capitalize()

    @field_validator("pdbx_stereo_config", mode="before")
    @staticmethod
    def _coerce_stereo_config(v: str | None) -> str | None:
        return None if v == "N" else v

    @field_validator("charge", mode="before")
    @staticmethod
    def _coerce_charge(v: str | None) -> str:
        return v or "0"


class ChemCompBond(LooseModel):
    atom_id_1: str
    atom_id_2: str
    value_order: int
    pdbx_aromatic_flag: bool = False
    pdbx_stereo_config: str | None = None

    @field_validator("value_order", mode="before")
    @staticmethod
    def _coerce_order(v: str | int) -> int:
        if isinstance(v, int):
            return v
        return {"sing": 1, "doub": 2, "trip": 3, "quad": 4}[v.lower()]

    @field_validator("pdbx_stereo_config", mode="before")
    @staticmethod
    def _coerce_stereo_config(v: str | None) -> str | None:
        return None if v == "N" else v


class ChemComp(LooseModel):
    id: str
    name: str
    type: str
    formula: str | None
    formula_weight: float | None
    mon_nstd_flag: bool | None = None

    atoms: list[ChemCompAtom]
    bonds: list[ChemCompBond]


class Scheme(LooseModel):
    asym_id: str
    entity_id: int
    mon_id: str

    seq_id: int = Field(
        validation_alias=AliasChoices(
            "seq_id",  # poly_seq_scheme
            "ndb_seq_num",  # nonpoly_scheme
            "num",  # branch_scheme
        )
    )

    pdb_seq_num: int | None  # this is auth_seq_id (!!!)
    pdb_ins_code: str | None = None

    def __eq__(self, other):
        if not isinstance(other, Scheme):
            return False

        return self.asym_id == other.asym_id and self.seq_id == other.seq_id

    def __lt__(self, other):
        if not isinstance(other, Scheme):
            return NotImplemented

        return (self.asym_id, self.seq_id) < (other.asym_id, other.seq_id)

    def __hash__(self):
        return hash((self.asym_id, self.seq_id))


class AtomSite(LooseModel):
    id: int
    type_symbol: str
    group_PDB: str

    label_atom_id: str
    label_alt_id: str | None
    label_comp_id: str
    label_asym_id: str
    label_seq_id: int | None

    auth_seq_id: int
    auth_comp_id: str
    auth_asym_id: str

    pdbx_PDB_ins_code: str | None
    pdbx_PDB_model_num: int

    cartn: NDArray[np.float64]
    occupancy: float

    @property
    def is_hydrogen(self):
        return self.type_symbol == "H"

    @model_validator(mode="before")
    @staticmethod
    def _gather_cartn(v: dict[str, Any]):
        v["cartn"] = np.array(
            [v["Cartn_x"], v["Cartn_y"], v["Cartn_z"]], dtype=np.float64
        )
        return v


class BioAssemblyGen(LooseModel):
    asym_id_list: list[str]
    operations: list[list[str]]

    @field_validator("asym_id_list", mode="before")
    @staticmethod
    def _parse_asym_ids(v: str):
        return v.split(",")

    @model_validator(mode="before")
    @staticmethod
    def _parse_operations(v: dict[str, str]):
        op_exprs = v["oper_expression"]

        if "(" not in op_exprs:
            v["operations"] = [  # type: ignore[assignment]
                _parse_expr(op_exprs)
            ]
            return v

        v["operations"] = [  # type: ignore[assignment]
            _parse_expr(ops)
            for token in op_exprs.split("(")
            if (ops := token.strip(")"))
        ]

        return v


def _parse_single_range(op: str) -> Iterable[str]:
    if "-" not in op:
        return [op]

    start, end = map(int, op.split("-"))
    return map(str, range(start, end + 1))


def _parse_expr(ops: str) -> list[str]:
    return [
        elem for expr in ops.split(",") for elem in _parse_single_range(expr)
    ]


class BioAssembly(LooseModel):
    id: int
    details: str
    oligomeric_details: str
    oligomeric_count: int

    assembly_gens: list[BioAssemblyGen]

    # XXX: 1hya has missing oligomeric_* fields. Assume monomeric for now.

    @field_validator("oligomeric_details", mode="before")
    @staticmethod
    def _parse_ol_detail(v: str | None):
        if v is None:
            return "monomeric (!MISSING!)"
        return v

    @field_validator("oligomeric_count", mode="before")
    @staticmethod
    def _parse_ol_cnt(v: str | None):
        if v is None:
            return 1
        return v


class SymOp(LooseModel):
    type: str
    name: str | None
    symmetry_operation: str | None

    matrix: NDArray[np.float64]
    vector: NDArray[np.float64]

    @model_validator(mode="before")
    @staticmethod
    def _gather_matrix_vector(v: dict[str, Any]):
        v["matrix"] = np.array(
            [
                [v[f"matrix[{i}][{j}]"] for j in range(1, 4)]
                for i in range(1, 4)
            ],
            dtype=np.float64,
        )
        v["vector"] = np.array(
            [v[f"vector[{i}]"] for i in range(1, 4)], dtype=np.float64
        )
        return v


class StructConnPartner(LooseModel):
    label_atom_id: str
    label_comp_id: str
    label_asym_id: str
    label_seq_id: int | None

    auth_seq_id: int
    auth_comp_id: str
    auth_asym_id: str

    pdbx_PDB_ins_code: str | None

    symmetry: str


class StructConn(LooseModel):
    id: str
    conn_type_id: str

    pdbx_leaving_atom_flag: int = Field(ge=0, le=2)
    pdbx_dist_value: float | None

    ptnr1: StructConnPartner
    ptnr2: StructConnPartner

    @model_validator(mode="before")
    @staticmethod
    def _gather_ptnrs(d: dict[str, Any]):
        for ptnr in ("ptnr1", "ptnr2"):
            d[ptnr] = {
                k.replace(f"{ptnr}_", ""): v
                for k, v in d.items()
                if f"{ptnr}_" in k
            }
        return d

    @field_validator("pdbx_leaving_atom_flag", mode="before")
    @staticmethod
    def _coerce_leaving_atom(v):
        return {"one": 1, "both": 2}.get(v, 0)


class StructConnType(LooseModel):
    criteria: str | None
    reference: str | None


class BranchLinkPartner(LooseModel):
    entity_branch_list_num: int
    comp_id: str
    atom_id: str
    leaving_atom_id: str


class BranchLink(LooseModel):
    value_order: int

    ptnr1: BranchLinkPartner
    ptnr2: BranchLinkPartner

    @model_validator(mode="before")
    @staticmethod
    def _gather_ptnrs(d: dict[str, Any]):
        for idx in range(1, 3):
            key = f"ptnr{idx}"
            suffix = f"_{idx}"
            d[key] = {
                k[: -len(suffix)]: v
                for k, v in d.items()
                if k.endswith(suffix)
            }
        return d

    @field_validator("value_order", mode="before")
    @staticmethod
    def _coerce_order(v: str) -> int:
        return {"sing": 1, "doub": 2, "trip": 3, "quad": 4}[v.lower()]


def _join_chem_comp(
    v: dict[str, Any],
) -> dict[str, dict[str, list[dict[str, str | None]]]]:
    chem_comps: dict[str, dict[str, list[dict[str, str | None]]]] = {
        cc["id"]: cc for cc in v.get("chem_comp", [])
    }
    for cc in chem_comps.values():
        cc["atoms"] = []
        cc["bonds"] = []

    for atom in v.get("chem_comp_atom", []):
        chem_comps[atom["comp_id"]]["atoms"].append(atom)

    for bond in v.get("chem_comp_bond", []):
        chem_comps[bond["comp_id"]]["bonds"].append(bond)

    return chem_comps


class PdbMetadata(BaseModel):
    entry_id: str
    exptl_method: str
    pdbx_keywords: str
    revision_date: dt.date
    resolution: float


class Mmcif(LooseModel):
    entry_id: str = Field(validation_alias=AliasPath("entry", 0, "id"))
    exptl_method: str = Field(validation_alias=AliasPath("exptl", 0, "method"))
    pdbx_keywords: str = Field(
        validation_alias=AliasPath("struct_keywords", 0, "pdbx_keywords")
    )

    revision_date: dt.date = Field(
        validation_alias=AliasPath(
            "pdbx_audit_revision_history", "revision_date"
        )
    )

    resolution: float = Field(default=999.9)

    entity: dict[int, Entity] = Field(default_factory=dict)

    pdbx_poly_seq_scheme: dict[str, list[Scheme]] = Field(default_factory=dict)
    pdbx_branch_scheme: dict[str, list[Scheme]] = Field(default_factory=dict)
    pdbx_nonpoly_scheme: dict[str, list[Scheme]] = Field(default_factory=dict)

    atom_site: list[AtomSite]

    pdbx_struct_assembly: list[BioAssembly]
    pdbx_struct_oper_list: dict[str, SymOp]
    struct_asym: dict[str, int]

    struct_conn: list[StructConn] = Field(default_factory=list)
    struct_conn_type: dict[str, StructConnType] = Field(default_factory=dict)

    pdbx_entity_branch_link: dict[int, list[BranchLink]] = Field(
        default_factory=dict
    )

    @field_validator("pdbx_keywords", mode="before")
    @staticmethod
    def _coerce_pdbx_kwds(v: Any) -> Any:
        if v is None:
            return ""
        return v

    @model_validator(mode="before")
    @staticmethod
    def _find_oldest(v: dict[str, list[dict[str, Any]]]):
        min_rev = min(
            (rev for rev in v.get("pdbx_audit_revision_history", [])),
            key=lambda r: int(r["ordinal"]),
        )
        v["pdbx_audit_revision_history"] = min_rev  # type: ignore[assignment]
        return v

    @model_validator(mode="before")
    @staticmethod
    def _join_bioassembly(v: dict[str, list[dict[str, Any]]]):
        bas = {ba["id"]: ba for ba in v.get("pdbx_struct_assembly", [])}
        for ba in bas.values():
            ba["assembly_gens"] = []

        for bag in v.get("pdbx_struct_assembly_gen", []):
            bas[bag["assembly_id"]]["assembly_gens"].append(bag)

        return v

    @model_validator(mode="before")
    @staticmethod
    def _find_resolution(v: dict[str, list[dict[str, Any]]]):
        for k1, k2 in (
            ("refine", "ls_d_res_high"),
            ("em_3d_reconstruction", "resolution"),
            ("reflns", "d_resolution_high"),
        ):
            try:
                res = v.get(k1, [])[0][k2]
            except (KeyError, IndexError):
                continue

            if res is not None:
                v["resolution"] = res
                break

        return v

    @model_validator(mode="before")
    @staticmethod
    def _join_branch_link(v: dict[str, list[dict[str, Any]]]):
        branch_links = defaultdict(list)
        for bl in v.get("pdbx_entity_branch_link", []):
            branch_links[bl["entity_id"]].append(bl)

        v["pdbx_entity_branch_link"] = branch_links  # type: ignore[assignment]

        return v

    @field_validator(
        "entity",
        "pdbx_struct_oper_list",
        "struct_conn_type",
        mode="before",
    )
    @staticmethod
    def _list_as_dict(v: list[dict[str, Any]]):
        return {d["id"]: d for d in v}

    @field_validator(
        "pdbx_poly_seq_scheme",
        "pdbx_branch_scheme",
        "pdbx_nonpoly_scheme",
        mode="before",
    )
    @staticmethod
    def _xform_schemes(v: list[dict[str, Any]]):
        ret = defaultdict(list)
        for d in v:
            ret[d["asym_id"]].append(d)
        return ret

    @field_validator("struct_asym", mode="before")
    @staticmethod
    def _xform_struct_asym(v: list[dict[str, Any]]):
        ret = {}
        for d in v:
            ret[d["id"]] = d["entity_id"]
        return ret

    def metadata(self) -> PdbMetadata:
        return PdbMetadata(
            entry_id=self.entry_id,
            exptl_method=self.exptl_method,
            pdbx_keywords=self.pdbx_keywords,
            revision_date=self.revision_date,
            resolution=self.resolution,
        )


def load_mmcif_single(file: Path):
    data = next(read_cif(file)).data
    mmcif = Mmcif.model_validate(cif_ddl2_frame_as_dict(data))
    return mmcif


def load_components(file: Path, max_count: int = 0):
    components: dict[str, ChemComp] = {}

    for i, block in tqdm(enumerate(read_cif(file))):
        data = block.data
        name = block.name

        try:
            mmcif_dict = cif_ddl2_frame_as_dict(data)
            chem_comp = _join_chem_comp(mmcif_dict)

            assert len(chem_comp) == 1
            components[name] = ChemComp.model_validate(chem_comp[name])
        except Exception:
            _logger.exception("Failed to load component %s", name)

        if max_count and i >= max_count - 1:
            break

    return components
