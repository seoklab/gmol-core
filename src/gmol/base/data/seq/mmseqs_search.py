"""
Some of the code is imported from ColabFold.
See:

* https://github.com/sokrypton/ColabFold/blob/main/colabfold/batch.py
* https://github.com/sokrypton/ColabFold/blob/main/colabfold/colabfold.py
* https://github.com/sokrypton/ColabFold/blob/main/colabfold/mmseqs/search.py
* https://github.com/sokrypton/ColabFold/blob/main/colabfold/utils.py


.. admonition:: License
    :collapsible: closed

    .. code-block::

        MIT License

        Copyright (c) 2021 Sergey Ovchinnikov

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.
"""

import itertools
import logging
import math
import os
import shutil
import subprocess as sp
import typing
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

from gmol.base.data.seq.colabfold_input import (
    MsaQuery,
    get_queries,
    msa_to_str,
)
from gmol.base.path import safe_filename

_logger = logging.getLogger(__name__)


class MMseqs:
    def __init__(
        self,
        threads: int = 1,
        skip_existing: bool = True,
        mmseqs: str | Path = "mmseqs",
    ):
        self.mmseqs = mmseqs
        self.threads = threads
        self.skip_existing = skip_existing

        self._env = os.environ.copy()
        self._env["MMSEQS_CALL_DEPTH"] = "1"

    def _run(
        self,
        params: Sequence[str | Path],
        output_path: Path | None = None,
        skip_existing: bool | None = None,
    ):
        skip_existing = (
            self.skip_existing if skip_existing is None else skip_existing
        )

        if skip_existing and output_path is not None and output_path.is_file():
            _logger.info(
                "Skipping %s because %s already exists", params[0], output_path
            )
            return

        _logger.info("Running %s: %s", self.mmseqs, params)
        sp.run([self.mmseqs, *params], check=True, env=self._env)

    def createdb(
        self,
        query_file: Path,
        qdb: Path,
        shuffle: int = 0,
        dbtype: int = 1,
    ):
        params: list[str | Path] = [
            "createdb",
            query_file,
            qdb,
            "--shuffle",
            str(shuffle),
        ]
        if dbtype != 0:
            params.extend(["--dbtype", str(dbtype)])
        self._run(params, output_path=qdb, skip_existing=False)

    def search(
        self,
        qdb: Path,
        db: Path,
        res: Path,
        tmp: Path,
        additional_search_param: Iterable[str | Path] | None = None,
    ):
        if additional_search_param is None:
            additional_search_param = []

        self._run(
            [
                "search",
                qdb,
                db,
                res,
                tmp,
                "--threads",
                str(self.threads),
                *additional_search_param,
            ],
            output_path=res,
        )

    def mvdb(self, src: Path, dest: Path):
        self._run(["mvdb", src, dest], output_path=dest)

    def rmdb(self, db: Path):
        self._run(["rmdb", db])

    def unpackdb(
        self,
        db: Path,
        dest: Path,
        unpack_name_mode: int = 0,
        unpack_suffix: str = ".a3m",
    ):
        self._run(
            [
                "unpackdb",
                db,
                dest,
                "--unpack-name-mode",
                str(unpack_name_mode),
                "--unpack-suffix",
                unpack_suffix,
            ],
            output_path=dest,
        )

    def result2msa(
        self,
        qdb: Path,
        db: Path,
        res: Path,
        dest: Path,
        msa_format_mode: int = 6,
        db_load_mode: int = 2,
        additional_params: Iterable[str | Path] | None = None,
    ):
        if additional_params is None:
            additional_params = []

        self._run(
            [
                "result2msa",
                qdb,
                db,
                res,
                dest,
                "--msa-format-mode",
                str(msa_format_mode),
                "--db-load-mode",
                str(db_load_mode),
                "--threads",
                str(self.threads),
                *additional_params,
            ],
            output_path=dest,
        )

    def filterresult(
        self,
        qdb: Path,
        db: Path,
        res: Path,
        dest: Path,
        db_load_mode: int = 2,
        qid: str = "0",
        qsc: float = 0.8,
        diff: int = 0,
        max_seq_id: float = 1.0,
        filter_min_enable: int = 100,
    ):
        self._run(
            [
                "filterresult",
                qdb,
                db,
                res,
                dest,
                "--db-load-mode",
                str(db_load_mode),
                "--qid",
                qid,
                "--qsc",
                str(qsc),
                "--diff",
                str(diff),
                "--max-seq-id",
                str(max_seq_id),
                "--threads",
                str(self.threads),
                "--filter-min-enable",
                str(filter_min_enable),
            ],
            output_path=dest,
        )

    def align(
        self,
        qdb: Path,
        db: Path,
        res: Path,
        dest: Path,
        db_load_mode: int = 2,
        align_eval: float | str = 10,
        max_accept: int = 1000000,
        alt_ali: int | None = 10,
        backtrace: bool = True,
    ):
        args: list[str | Path] = [
            "align",
            qdb,
            db,
            res,
            dest,
            "--db-load-mode",
            str(db_load_mode),
            "-e",
            str(align_eval),
            "--max-accept",
            str(max_accept),
            "--threads",
            str(self.threads),
        ]
        if alt_ali is not None:
            args += ["--alt-ali", str(alt_ali)]
        if backtrace:
            args.append("-a")

        self._run(args, output_path=dest)

    def expandaln(
        self,
        qdb: Path,
        db: Path,
        res: Path,
        db2: Path,
        dest: Path,
        db_load_mode: int = 2,
        additional_expand_param: Iterable[str | Path] | None = None,
    ):
        if additional_expand_param is None:
            additional_expand_param = []

        self._run(
            [
                "expandaln",
                qdb,
                db,
                res,
                db2,
                dest,
                "--db-load-mode",
                str(db_load_mode),
                "--threads",
                str(self.threads),
                *additional_expand_param,
            ],
            output_path=dest,
        )

    def lndb(self, src: Path, dest: Path):
        self._run(["lndb", src, dest], output_path=dest)

    def mergedbs(self, qdb: Path, dest: Path, *dbs: Path):
        self._run(["mergedbs", qdb, dest, *dbs], output_path=dest)

    def convertalis(
        self,
        qdb: Path,
        db: Path,
        res: Path,
        dest: Path,
        db_load_mode: int = 2,
        format_output: str = (
            "query,target,fident,alnlen,mismatch,gapopen,"
            "qstart,qend,tstart,tend,evalue,bits,cigar"
        ),
        db_output: int = 1,
    ):
        """Convert alignment result to tabular or DB. Skip if dest.dbtype exists."""
        skip_path = Path(str(dest) + ".dbtype")
        self._run(
            [
                "convertalis",
                qdb,
                db,
                res,
                dest,
                "--format-output",
                format_output,
                "--db-output",
                str(db_output),
                "--db-load-mode",
                str(db_load_mode),
                "--threads",
                str(self.threads),
            ],
            output_path=skip_path,
        )

    def pairaln(
        self,
        qdb: Path,
        db: Path,
        res: Path,
        dest: Path,
        db_load_mode: int = 2,
        pairing_mode: int = 0,
        pairing_dummy_mode: int = 0,
    ):
        self._run(
            [
                "pairaln",
                qdb,
                db,
                res,
                dest,
                "--db-load-mode",
                str(db_load_mode),
                "--pairing-mode",
                str(pairing_mode),
                "--pairing-dummy-mode",
                str(pairing_dummy_mode),
                "--threads",
                str(self.threads),
            ],
            output_path=dest,
        )


@dataclass
class MMseqsQuery:
    id: str
    unique_seqs: list[str]
    seq_counts: list[int]

    @classmethod
    def from_msa_query(cls, msa_query: MsaQuery) -> "MMseqsQuery":
        query_seqs_counts = Counter(msa_query.seqs)
        query_seqs_unique = list(query_seqs_counts.keys())
        seq_cnt = list(query_seqs_counts.values())
        return cls(msa_query.id, query_seqs_unique, seq_cnt)


@dataclass(kw_only=True)
class MMseqsCommonSearchParams:
    prefilter_mode: int = 0
    s: float | None = 8
    gpu: int = 0
    db_load_mode: int = 2
    ignore_index: bool = True

    @property
    def search_param(self):
        param = [
            "--num-iterations",
            "3",
            "--db-load-mode",
            str(self.db_load_mode),
            "-a",
            "-e",
            "0.1",
            "--max-seqs",
            "10000",
        ]
        if self.gpu:
            # gpu version only supports ungapped prefilter currently
            param += [
                "--gpu",
                str(self.gpu),
                "--prefilter-mode",
                "1",
            ]
        else:
            param += ["--prefilter-mode", str(self.prefilter_mode)]
            if self.s is not None:
                param += ["-s", f"{self.s:.1f}"]
            else:
                param += ["--k-score", "'seq:96,prof:80'"]
        return param


@dataclass(kw_only=True)
class MMseqsMonomerSearchParams(MMseqsCommonSearchParams):
    uniref_db: Path
    metagenomic_db: Path | None
    template_db: Path | None

    expand_eval: float = math.inf
    align_eval: int = 10
    diff: int = 3000
    qsc: float = -20.0
    max_accept: int = 1000000
    template_s: float | None = 7.5
    alt_ali: int = 10

    limit: bool = True

    suffix1: str = field(init=False)
    suffix2: str = field(init=False)
    suffix3: str = field(init=False)

    @property
    def template_search_param(self):
        param = []
        if self.gpu:
            # gpu version only supports ungapped prefilter currently
            param += [
                "--gpu",
                str(self.gpu),
                "--prefilter-mode",
                "1",
            ]
        else:
            param += ["--prefilter-mode", str(self.prefilter_mode)]
            if self.template_s is not None:
                param += ["-s", f"{self.template_s:.1f}"]
        return param

    @property
    def filter_param(self):
        return [
            "--filter-msa",
            str(int(self.limit)),
            "--filter-min-enable",
            "1000",
            "--diff",
            str(self.diff),
            "--qid",
            "0.0,0.2,0.4,0.6,0.8,1.0",
            "--qsc",
            "0",
            "--max-seq-id",
            "0.95",
        ]

    @property
    def expand_param(self):
        return [
            "--expansion-mode",
            "0",
            "-e",
            str(self.expand_eval),
            "--expand-filter-clusters",
            str(int(self.limit)),
            "--max-seq-id",
            "0.95",
        ]

    @property
    def udb1(self) -> Path:
        return self.uniref_db.with_name(f"{self.uniref_db.name}{self.suffix1}")

    @property
    def udb2(self) -> Path:
        return self.uniref_db.with_name(f"{self.uniref_db.name}{self.suffix2}")

    @property
    def mdb1(self) -> Path:
        mdb = typing.cast(Path, self.metagenomic_db)
        return mdb.with_name(f"{mdb.name}{self.suffix1}")

    @property
    def mdb2(self) -> Path:
        mdb = typing.cast(Path, self.metagenomic_db)
        return mdb.with_name(f"{mdb.name}{self.suffix2}")

    @property
    def tdb(self) -> Path:
        tdb = typing.cast(Path, self.template_db)
        return tdb.with_name(f"{tdb.name}{self.suffix3}")

    def __post_init__(self):
        if self.limit:
            self.align_eval = 10
            self.qsc = 0.8
            self.max_accept = 100000

        used_dbs = [self.uniref_db]
        if self.template_db is not None:
            used_dbs.append(self.template_db)
        if self.metagenomic_db is not None:
            used_dbs.append(self.metagenomic_db)

        for db in used_dbs:
            if not db.with_name(f"{db.name}.dbtype").is_file():
                raise FileNotFoundError(f"Database {db} does not exist")

            if self.ignore_index or (
                not db.with_name(f"{db.name}.idx").is_file()
                and not db.with_name(f"{db.name}.idx.index").is_file()
            ):
                _logger.info("Search does not use index")
                self.db_load_mode = 0
                self.suffix1 = "_seq"
                self.suffix2 = "_aln"
                self.suffix3 = ""
            else:
                self.suffix1 = ".idx"
                self.suffix2 = ".idx"
                self.suffix3 = ".idx"


@dataclass(kw_only=True)
class MMseqsPairSearchParams(MMseqsCommonSearchParams):
    search_db: Path

    pairing_strategy: int = 0

    suffix1: str = field(init=False)
    suffix2: str = field(init=False)

    @property
    def expand_param(self):
        return [
            "--expansion-mode",
            "0",
            "-e",
            "inf",
            "--expand-filter-clusters",
            "0",
            "--max-seq-id",
            "0.95",
        ]

    @property
    def udb1(self) -> Path:
        return self.search_db.with_name(f"{self.search_db.name}{self.suffix1}")

    @property
    def udb2(self) -> Path:
        return self.search_db.with_name(f"{self.search_db.name}{self.suffix2}")

    def __post_init__(self):
        if not self.search_db.with_name(
            f"{self.search_db.name}.dbtype"
        ).is_file():
            raise FileNotFoundError(
                f"Database {self.search_db} does not exist"
            )

        if self.ignore_index or (
            not self.search_db.with_name(
                f"{self.search_db.name}.idx"
            ).is_file()
            and not self.search_db.with_name(
                f"{self.search_db.name}.idx.index"
            ).is_file()
        ):
            _logger.info("Search does not use index")
            self.db_load_mode = 0
            self.suffix1 = "_seq"
            self.suffix2 = "_aln"
        else:
            self.suffix1 = ".idx"
            self.suffix2 = ".idx"


def mmseqs_search_monomer(
    mmseqs: MMseqs,
    qdb: Path,
    output_dir: Path,
    params: MMseqsMonomerSearchParams,
):
    """
    Run mmseqs with a local colabfold database set

    * db1: uniprot db (UniRef30)
    * db3: metagenomic db (colabfold_envdb_202108 or bfd_mgy_colabfold,
      the former is preferred)
    """
    p = params

    output_dir.mkdir(exist_ok=True, parents=True)

    with TemporaryDirectory() as tmpd:
        base = Path(tmpd)

        mmseqs.search(
            qdb,
            p.uniref_db,
            base / "res",
            base / "tmp",
            additional_search_param=p.search_param,
        )
        mmseqs.mvdb(base / "tmp/latest/profile_1", base / "prof_res")
        mmseqs.lndb(qdb.parent / f"{qdb.name}_h", base / "prof_res_h")
        mmseqs.expandaln(
            qdb,
            p.udb1,
            base / "res",
            p.udb2,
            base / "res_exp",
            db_load_mode=p.db_load_mode,
            additional_expand_param=p.expand_param,
        )
        mmseqs.align(
            base / "prof_res",
            p.udb1,
            base / "res_exp",
            base / "res_exp_realign",
            db_load_mode=p.db_load_mode,
            align_eval=p.align_eval,
            max_accept=p.max_accept,
            alt_ali=p.alt_ali,
        )
        mmseqs.filterresult(
            qdb,
            p.udb1,
            base / "res_exp_realign",
            base / "res_exp_realign_filter",
            db_load_mode=p.db_load_mode,
            qid="0",
            qsc=p.qsc,
            diff=0,
            max_seq_id=1.0,
            filter_min_enable=100,
        )
        mmseqs.result2msa(
            qdb,
            p.udb1,
            base / "res_exp_realign_filter",
            base / "uniref.a3m",
            msa_format_mode=6,
            db_load_mode=p.db_load_mode,
            additional_params=p.filter_param,
        )
        mmseqs.rmdb(base / "res_exp_realign_filter")
        mmseqs.rmdb(base / "res_exp_realign")
        mmseqs.rmdb(base / "res_exp")
        mmseqs.rmdb(base / "res")

        if p.metagenomic_db is not None:
            mmseqs.search(
                base / "prof_res",
                p.metagenomic_db,
                base / "res_env",
                base / "tmp3",
                additional_search_param=p.search_param,
            )

            additional_expand_param = [
                "--expansion-mode",
                "0",
                "-e",
                str(p.expand_eval),
            ]
            mmseqs.expandaln(
                base / "prof_res",
                p.mdb1,
                base / "res_env",
                p.mdb2,
                base / "res_env_exp",
                db_load_mode=p.db_load_mode,
                additional_expand_param=additional_expand_param,
            )
            mmseqs.align(
                base / "tmp3/latest/profile_1",
                p.mdb1,
                base / "res_env_exp",
                base / "res_env_exp_realign",
                db_load_mode=p.db_load_mode,
                align_eval=p.align_eval,
                max_accept=p.max_accept,
                alt_ali=p.alt_ali,
            )
            mmseqs.filterresult(
                qdb,
                p.mdb1,
                base / "res_env_exp_realign",
                base / "res_env_exp_realign_filter",
                db_load_mode=p.db_load_mode,
                qid="0",
                qsc=p.qsc,
                diff=0,
                max_seq_id=1.0,
                filter_min_enable=100,
            )
            mmseqs.result2msa(
                qdb,
                p.mdb1,
                base / "res_env_exp_realign_filter",
                base / "bfd.mgnify30.metaeuk30.smag30.a3m",
                msa_format_mode=6,
                db_load_mode=p.db_load_mode,
                additional_params=p.filter_param,
            )
            mmseqs.rmdb(base / "res_env_exp_realign_filter")
            mmseqs.rmdb(base / "res_env_exp_realign")
            mmseqs.rmdb(base / "res_env_exp")
            mmseqs.rmdb(base / "res_env")

        if p.template_db is not None:
            mmseqs.search(
                base / "prof_res",
                p.template_db,
                base / "res_pdb",
                base / "tmp2",
                additional_search_param=[
                    "--db-load-mode",
                    str(p.db_load_mode),
                    "-a",
                    "-e",
                    "0.1",
                    *p.template_search_param,
                ],
            )
            mmseqs.convertalis(
                base / "prof_res",
                p.tdb,
                base / "res_pdb",
                base / p.template_db.name,
                db_load_mode=p.db_load_mode,
            )
            mmseqs.rmdb(base / "res_pdb")

        if p.metagenomic_db is not None:
            mmseqs.mergedbs(
                qdb,
                base / "final.a3m",
                base / "uniref.a3m",
                base / "bfd.mgnify30.metaeuk30.smag30.a3m",
            )
        else:
            mmseqs.mvdb(base / "uniref.a3m", base / "final.a3m")

        mmseqs.unpackdb(base / "final.a3m", output_dir)
        if p.template_db is not None:
            mmseqs.unpackdb(
                base / p.template_db.name,
                output_dir,
                unpack_suffix=".m8",
            )


def mmseqs_search_pair(
    mmseqs: MMseqs,
    qdb: Path,
    output_dir: Path,
    params: MMseqsPairSearchParams,
    unpack_suffix: str,
) -> None:
    p = params

    with TemporaryDirectory() as tmpd:
        base = Path(tmpd)

        mmseqs.search(
            qdb,
            p.search_db,
            base / "res",
            base / "tmp",
            additional_search_param=p.search_param,
        )
        mmseqs.mvdb(base / "tmp/latest/profile_1", base / "prof_res")
        mmseqs.lndb(qdb.parent / f"{qdb.name}_h", base / "prof_res_h")
        mmseqs.expandaln(
            qdb,
            p.udb1,
            base / "res",
            p.udb2,
            base / "res_exp",
            db_load_mode=p.db_load_mode,
            additional_expand_param=p.expand_param,
        )
        mmseqs.align(
            base / "prof_res",
            p.udb1,
            base / "res_exp",
            base / "res_exp_realign",
            db_load_mode=p.db_load_mode,
            align_eval=0.001,
            max_accept=1000000,
            alt_ali=None,
            backtrace=False,
        )
        mmseqs.pairaln(
            qdb,
            p.search_db,
            base / "res_exp_realign",
            base / "res_exp_realign_pair",
            db_load_mode=p.db_load_mode,
            pairing_mode=p.pairing_strategy,
            pairing_dummy_mode=0,
        )
        mmseqs.align(
            base / "prof_res",
            p.udb1,
            base / "res_exp_realign_pair",
            base / "res_exp_realign_pair_bt",
            db_load_mode=p.db_load_mode,
            align_eval="inf",
            max_accept=1000000,
            alt_ali=None,
        )
        mmseqs.pairaln(
            qdb,
            p.search_db,
            base / "res_exp_realign_pair_bt",
            base / "res_final",
            db_load_mode=p.db_load_mode,
            pairing_mode=p.pairing_strategy,
            pairing_dummy_mode=1,
        )
        mmseqs.result2msa(
            qdb,
            p.udb1,
            base / "res_final",
            base / "pair.a3m",
            msa_format_mode=5,
            db_load_mode=p.db_load_mode,
        )

        mmseqs.unpackdb(
            base / "pair.a3m",
            output_dir,
            unpack_suffix=unpack_suffix,
        )


def run_search_from_path(
    query: Path,
    output_dir: Path,
    threads: int = 1,
    mmseqs: str | Path = "mmseqs",
    *,
    monomer_params: MMseqsMonomerSearchParams,
    pair_params: MMseqsPairSearchParams,
    env_pair_params: MMseqsPairSearchParams | None = None,
) -> None:
    """Run MSA search from a FASTA/CSV/dir path. Results written as
    output_dir/<jobname>.a3m and output_dir/<jobname>_<template_db>.m8
    (if template search is enabled).

    For complex (multi-chain) input, runs monomer unpaired + optional paired
    search and merges per job.
    """
    queries = get_queries(query, None)
    if not queries:
        raise ValueError(f"No queries found in {query}")

    queries_unique: list[MMseqsQuery] = [
        MMseqsQuery.from_msa_query(q) for q in queries
    ]

    output_dir.mkdir(exist_ok=True, parents=True)
    query_fas = output_dir / "query.fas"
    with query_fas.open("w") as f:
        for q in queries_unique:
            for j, seq in enumerate(q.unique_seqs, start=101):
                f.write(f">{j}\n{seq}\n")

    runner = MMseqs(threads=threads, mmseqs=mmseqs)
    runner.createdb(query_fas, output_dir / "qdb", shuffle=0)

    with (output_dir / "qdb.lookup").open("w") as f:
        seqid = itertools.count(0)
        for fid, q in enumerate(queries_unique):
            for _ in q.unique_seqs:
                raw_first_id = q.id.split()[0]
                f.write(f"{next(seqid)}\t{raw_first_id}\t{fid}\n")

    mmseqs_search_monomer(
        runner,
        output_dir / "qdb",
        output_dir,
        monomer_params,
    )

    if any(q.is_complex for q in queries):
        mmseqs_search_pair(
            runner,
            output_dir / "qdb",
            output_dir,
            pair_params,
            ".paired.a3m",
        )
        if env_pair_params is not None:
            mmseqs_search_pair(
                runner,
                output_dir / "qdb",
                output_dir,
                env_pair_params,
                ".env.paired.a3m",
            )

        # Merge per-job: read unpaired/paired a3m by index, msa_to_str, write job_index.a3m
        seqid = itertools.count(0)
        for qid, q in enumerate(queries_unique):
            unpaired_msa: list[str] = []
            paired_msa: list[str] = []
            heteromer = len(q.unique_seqs) > 1
            for _ in q.unique_seqs:
                sid = next(seqid)

                a3m_path = output_dir / f"{sid}.a3m"
                unpaired_msa.append(a3m_path.read_text())
                a3m_path.unlink()

                paired_path = output_dir / f"{sid}.paired.a3m"
                if env_pair_params is not None:
                    env_paired_path = output_dir / f"{sid}.env.paired.a3m"
                    if heteromer:
                        with (
                            open(env_paired_path) as fin,
                            open(paired_path, "a") as fout,
                        ):
                            shutil.copyfileobj(fin, fout)
                    env_paired_path.unlink()
                if heteromer:
                    paired_msa.append(paired_path.read_text())
                paired_path.unlink()

            msa = msa_to_str(
                unpaired_msa,
                paired_msa if heteromer else None,
                q.unique_seqs,
                q.seq_counts,
            )
            (output_dir / f"{qid}.a3m").write_text(msa)

    seqid = itertools.count(0)
    for qid, q in enumerate(queries_unique):
        out_key = safe_filename(q.id)

        output_dir.joinpath(f"{qid}.a3m").rename(output_dir / f"{out_key}.a3m")

        if monomer_params.template_db is not None:
            templates: list[Path] = []
            for _ in q.unique_seqs:
                sid = next(seqid)
                templates.append(output_dir / f"{sid}.m8")

            with (
                output_dir / f"{out_key}_{monomer_params.template_db.stem}.m8"
            ).open("w") as fout:
                for t in templates:
                    with t.open() as fin:
                        shutil.copyfileobj(fin, fout)
                    t.unlink()

    runner.rmdb(output_dir / "qdb")
    runner.rmdb(output_dir / "qdb_h")
    query_fas.unlink(missing_ok=True)
