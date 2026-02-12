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

import logging
import math
import os
import random
import subprocess as sp
from collections.abc import Iterable, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import pandas as pd

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
    ):
        if (
            self.skip_existing
            and output_path is not None
            and output_path.is_file()
        ):
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
    ):
        self._run(
            ["createdb", query_file, qdb, "--shuffle", str(shuffle)],
            output_path=qdb,
        )

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
        align_eval: int = 10,
        max_accept: int = 1000000,
        alt_ali: int = 10,
    ):
        self._run(
            [
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
                "--alt-ali",
                str(alt_ali),
                "-a",
            ],
            output_path=dest,
        )

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


def get_queries(
    file_path: Path, sort_by: Literal["length", "random"] = "length"
) -> list[tuple[str, list[str]]]:
    """Loads sequences from a specified file (CSV, TSV, or FASTA) and returns
    a list of tuples containing job name, sequence(s).
    The sequences can be sorted by length or shuffled randomly.
    """
    if not file_path.is_file():
        raise FileNotFoundError(f"File '{file_path}' could not be found.")

    sep = "\t" if file_path.suffix == ".tsv" else ","
    df = pd.read_csv(file_path, sep=sep)
    if "id" not in df.columns or "sequence" not in df.columns:
        raise ValueError("The file must contain 'id' and 'sequence' columns.")

    sequences: list[tuple[str, list[str]]] = [
        (seq_id, sequence.upper().split(":"))
        for seq_id, sequence in df[["id", "sequence"]].itertuples(index=False)
    ]

    if sort_by == "length":
        sequences.sort(key=lambda seqs: len("".join(seqs[1])))
    elif sort_by == "random":
        random.shuffle(sequences)
    else:
        raise ValueError("Invalid sort_by value. Use 'length' or 'random'.")

    _logger.info("Loaded %d sequences from %s", len(sequences), file_path)
    return sequences


def mmseqs_search_monomer(
    mmseqs: MMseqs,
    qdb: Path,
    output_dir: Path,
    uniref_db: Path,
    metagenomic_db: Path | None,
    limit: bool = True,
    expand_eval: float = math.inf,
    align_eval: int = 10,
    diff: int = 3000,
    qsc: float = -20.0,
    max_accept: int = 1000000,
    prefilter_mode: int = 0,
    s: float | None = 8,
    db_load_mode: int = 2,
    alt_ali: int = 10,
):
    """
    Run mmseqs with a local colabfold database set

    * db1: uniprot db (UniRef30)
    * db3: metagenomic db (colabfold_envdb_202108 or bfd_mgy_colabfold,
      the former is preferred)
    """
    used_dbs = [uniref_db]
    if metagenomic_db is not None:
        used_dbs.append(metagenomic_db)

    for db in used_dbs:
        if not db.with_name(f"{db.name}.dbtype").is_file():
            raise FileNotFoundError(f"Database {db} does not exist")

        if (
            not db.with_name(f"{db.name}.idx").is_file()
            and not db.with_name(f"{db.name}.idx.index").is_file()
        ) or os.environ.get("MMSEQS_IGNORE_INDEX", ""):
            _logger.info("Search does not use index")
            db_load_mode = 0
            suffix1 = "_seq"
            suffix2 = "_aln"
        else:
            suffix1 = ".idx"
            suffix2 = ".idx"

    output_dir.mkdir(exist_ok=True, parents=True)

    if limit:
        # 0.1 was not used in benchmarks due to POSIX shell bug in line above
        #  EXPAND_EVAL=0.1
        align_eval = 10
        qsc = 0.8
        max_accept = 100000

    search_param = [
        "--num-iterations",
        "3",
        "--db-load-mode",
        str(db_load_mode),
        "-a",
        "-e",
        "0.1",
        "--max-seqs",
        "10000",
        "--prefilter-mode",
        str(prefilter_mode),
    ]
    if s is not None:
        search_param += ["-s", f"{s:.1f}"]

    filter_param = [
        "--filter-msa",
        str(int(limit)),
        "--filter-min-enable",
        "1000",
        "--diff",
        str(diff),
        "--qid",
        "0.0,0.2,0.4,0.6,0.8,1.0",
        "--qsc",
        "0",
        "--max-seq-id",
        "0.95",
    ]
    expand_param = [
        "--expansion-mode",
        "0",
        "-e",
        str(expand_eval),
        "--expand-filter-clusters",
        str(int(limit)),
        "--max-seq-id",
        "0.95",
    ]

    with TemporaryDirectory() as tmpd:
        base = Path(tmpd)

        udb1 = uniref_db.with_name(f"{uniref_db.name}{suffix1}")
        udb2 = uniref_db.with_name(f"{uniref_db.name}{suffix2}")

        mmseqs.search(
            qdb,
            uniref_db,
            base / "res",
            base / "tmp",
            additional_search_param=search_param,
        )
        mmseqs.mvdb(base / "tmp/latest/profile_1", base / "prof_res")
        mmseqs.lndb(base / "pdb_h", base / "prof_res_h")
        mmseqs.expandaln(
            qdb,
            udb1,
            base / "res",
            udb2,
            base / "res_exp",
            db_load_mode=db_load_mode,
            additional_expand_param=expand_param,
        )
        mmseqs.align(
            base / "prof_res",
            udb1,
            base / "res_exp",
            base / "res_exp_realign",
            db_load_mode=db_load_mode,
            align_eval=align_eval,
            max_accept=max_accept,
            alt_ali=alt_ali,
        )
        mmseqs.filterresult(
            qdb,
            udb1,
            base / "res_exp_realign",
            base / "res_exp_realign_filter",
            db_load_mode=db_load_mode,
            qid="0",
            qsc=qsc,
            diff=0,
            max_seq_id=1.0,
            filter_min_enable=100,
        )
        mmseqs.result2msa(
            qdb,
            udb1,
            base / "res_exp_realign_filter",
            base / "uniref.a3m",
            msa_format_mode=6,
            db_load_mode=db_load_mode,
            additional_params=filter_param,
        )
        mmseqs.rmdb(base / "res_exp_realign_filter")
        mmseqs.rmdb(base / "res_exp_realign")
        mmseqs.rmdb(base / "res_exp")
        mmseqs.rmdb(base / "res")

        if metagenomic_db is not None:
            mdb1 = metagenomic_db.with_name(f"{metagenomic_db.name}{suffix1}")
            mdb2 = metagenomic_db.with_name(f"{metagenomic_db.name}{suffix2}")

            mmseqs.search(
                base / "prof_res",
                metagenomic_db,
                base / "res_env",
                base / "tmp3",
                additional_search_param=search_param,
            )

            additional_expand_param = [
                "--expansion_mode",
                "0",
                "--e",
                str(expand_eval),
            ]
            mmseqs.expandaln(
                qdb,
                mdb1,
                base / "res_env",
                mdb2,
                base / "res_env_exp",
                db_load_mode=db_load_mode,
                additional_expand_param=additional_expand_param,
            )
            mmseqs.align(
                base / "tmp3/latest/profile_1",
                mdb1,
                base / "res_env_exp",
                base / "res_env_exp_realign",
                db_load_mode=db_load_mode,
                align_eval=align_eval,
                max_accept=max_accept,
                alt_ali=alt_ali,
            )
            mmseqs.filterresult(
                qdb,
                mdb1,
                base / "res_env_exp_realign",
                base / "res_env_exp_realign_filter",
                db_load_mode=db_load_mode,
                qid="0",
                qsc=qsc,
                diff=0,
                max_seq_id=1.0,
                filter_min_enable=100,
            )
            mmseqs.result2msa(
                qdb,
                mdb1,
                base / "res_env_exp_realign_filter",
                base / "bfd.mgnify30.metaeuk30.smag30.a3m",
                msa_format_mode=6,
                db_load_mode=db_load_mode,
                additional_params=filter_param,
            )
            mmseqs.rmdb(base / "res_env_exp_realign_filter")
            mmseqs.rmdb(base / "res_env_exp_realign")
            mmseqs.rmdb(base / "res_env_exp")
            mmseqs.rmdb(base / "res_env")

        mmseqs.mergedbs(qdb, base / "final.a3m", *used_dbs)
        mmseqs.unpackdb(base / "final.a3m", output_dir)
