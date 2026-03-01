r"""
ColabFold-style MMseqs2 MSA search (library only).

Use ``MMseqs``, ``mmseqs_search_monomer``, ``mmseqs_search_pair``, ``get_queries``
for programmatic runs. Optional ``base``, ``template_db``, ``use_templates``
for persistent work dir and template search.

**FASTA (or CSV/dir) input in one call:** use ``run_search_from_path`` (supports
monomer and complex/paired)::

    from pathlib import Path
    from gmol.base.data.seq.mmseqs_search import MMseqs, run_search_from_path

    run_search_from_path(
        query_path=Path("queries.fasta"),
        dbbase=Path("/data/colabfold_db"),
        output_dir=Path("msas"),
        uniref_db=Path("uniref30_2302_db"),   # or dbbase / "uniref30_2302_db"
        metagenomic_db=Path("colabfold_envdb_202108_db"),
        threads=8,
    )
    # Complex (A:B): set use_env_pairing=True for env pairing; pair_mode="unpaired"|"paired"|"unpaired_paired".
    # Writes msas/<jobname>.a3m per query (jobname from FASTA header).

**Step-by-step (FASTA):** parse with ``get_queries_from_path``, write query.fas,
then ``createdb`` + ``mmseqs_search_monomer``::

    from gmol.base.data.seq.colabfold_input import get_queries_from_path
    from gmol.base.data.seq.mmseqs_search import MMseqs, mmseqs_search_monomer

    queries, is_complex = get_queries_from_path(Path("in.fasta"))
    base = Path("work")
    base.mkdir(exist_ok=True)
    with (base / "query.fas").open("w") as f:
        for i, (_, seq, _, _) in enumerate(queries):
            s = seq if isinstance(seq, str) else seq[0]
            f.write(f">{101 + i}\n{s}\n")
    mm = MMseqs(threads=8)
    mm.createdb(base / "query.fas", base / "qdb", shuffle=0, dbtype=1)
    mmseqs_search_monomer(mm, base / "qdb", base, uniref_db, metagenomic_db, base=base)
    # Results in base/*.a3m (one per sequence in query.fas by index).

For FASTA/CSV/dir and complex A:B parsing, ``get_queries_from_path`` lives in
``gmol.base.data.seq.colabfold_input``.

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
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import pandas as pd

from gmol.base.data.seq.colabfold_input import (
    get_queries_from_path,
    msa_to_str,
    safe_filename,
)

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
        dbtype: int = 1,
    ):
        params = ["createdb", query_file, qdb, "--shuffle", str(shuffle)]
        if dbtype != 0:
            params.extend(["--dbtype", str(dbtype)])
        self._run(params, output_path=qdb)

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

    def convertalis(
        self,
        qdb: Path,
        db: Path,
        res: Path,
        dest: Path,
        db_load_mode: int = 2,
        format_output: str = "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,cigar",
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
    *,
    base: Path | None = None,
    template_db: Path | None = None,
    use_templates: bool = False,
    gpu: int = 0,
    gpu_server: int = 0,
):
    """
    Run mmseqs with a local colabfold database set

    * db1: uniprot db (UniRef30)
    * db3: metagenomic db (colabfold_envdb_202108 or bfd_mgy_colabfold,
      the former is preferred)

    If base is provided, use it as the work directory and unpack there;
    otherwise use a temporary directory and unpack to output_dir.
    If use_templates and template_db are set, run template search and
    convertalis to produce template_db.m8.
    """
    used_dbs = [uniref_db]
    if metagenomic_db is not None:
        used_dbs.append(metagenomic_db)
    if use_templates and template_db is not None:
        used_dbs.append(template_db)

    used_dbs_for_merge = [uniref_db]
    if metagenomic_db is not None:
        used_dbs_for_merge.append(metagenomic_db)

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
            suffix3 = ""
        else:
            suffix1 = ".idx"
            suffix2 = ".idx"
            suffix3 = ".idx"

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
    if gpu:
        search_param = [
            *search_param,
            "--gpu",
            str(gpu),
            "--prefilter-mode",
            "1",
        ]
    if gpu_server:
        search_param = [*search_param, "--gpu-server", str(gpu_server)]
    if s is not None:
        search_param += ["-s", f"{s:.1f}"]

    template_search_param: list[str | Path] = []
    if use_templates and template_db is not None:
        if gpu:
            template_search_param = [
                "--gpu",
                str(gpu),
                "--prefilter-mode",
                "1",
            ]
        else:
            template_search_param = [
                "-s",
                "7.5",
                "--prefilter-mode",
                str(prefilter_mode),
            ]

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

    @contextmanager
    def _work_dir() -> Iterable[tuple[Path, Path]]:
        if base is not None:
            base.mkdir(exist_ok=True, parents=True)
            yield base, base
        else:
            with TemporaryDirectory() as tmpd:
                yield Path(tmpd), output_dir

    with _work_dir() as (work_base, unpack_dest):
        udb1 = uniref_db.with_name(f"{uniref_db.name}{suffix1}")
        udb2 = uniref_db.with_name(f"{uniref_db.name}{suffix2}")

        mmseqs.search(
            qdb,
            uniref_db,
            work_base / "res",
            work_base / "tmp",
            additional_search_param=search_param,
        )
        mmseqs.mvdb(work_base / "tmp/latest/profile_1", work_base / "prof_res")
        mmseqs.lndb(qdb.with_name("qdb_h"), work_base / "prof_res_h")
        mmseqs.expandaln(
            qdb,
            udb1,
            work_base / "res",
            udb2,
            work_base / "res_exp",
            db_load_mode=db_load_mode,
            additional_expand_param=expand_param,
        )
        mmseqs.align(
            work_base / "prof_res",
            udb1,
            work_base / "res_exp",
            work_base / "res_exp_realign",
            db_load_mode=db_load_mode,
            align_eval=align_eval,
            max_accept=max_accept,
            alt_ali=alt_ali,
        )
        mmseqs.filterresult(
            qdb,
            udb1,
            work_base / "res_exp_realign",
            work_base / "res_exp_realign_filter",
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
            work_base / "res_exp_realign_filter",
            work_base / "uniref.a3m",
            msa_format_mode=6,
            db_load_mode=db_load_mode,
            additional_params=filter_param,
        )
        mmseqs.rmdb(work_base / "res_exp_realign_filter")
        mmseqs.rmdb(work_base / "res_exp_realign")
        mmseqs.rmdb(work_base / "res_exp")
        mmseqs.rmdb(work_base / "res")

        if metagenomic_db is not None:
            mdb1 = metagenomic_db.with_name(f"{metagenomic_db.name}{suffix1}")
            mdb2 = metagenomic_db.with_name(f"{metagenomic_db.name}{suffix2}")

            mmseqs.search(
                work_base / "prof_res",
                metagenomic_db,
                work_base / "res_env",
                work_base / "tmp3",
                additional_search_param=search_param,
            )

            additional_expand_param = [
                "--expansion-mode",
                "0",
                "-e",
                str(expand_eval),
            ]
            mmseqs.expandaln(
                qdb,
                mdb1,
                work_base / "res_env",
                mdb2,
                work_base / "res_env_exp",
                db_load_mode=db_load_mode,
                additional_expand_param=additional_expand_param,
            )
            mmseqs.align(
                work_base / "tmp3/latest/profile_1",
                mdb1,
                work_base / "res_env_exp",
                work_base / "res_env_exp_realign",
                db_load_mode=db_load_mode,
                align_eval=align_eval,
                max_accept=max_accept,
                alt_ali=alt_ali,
            )
            mmseqs.filterresult(
                qdb,
                mdb1,
                work_base / "res_env_exp_realign",
                work_base / "res_env_exp_realign_filter",
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
                work_base / "res_env_exp_realign_filter",
                work_base / "bfd.mgnify30.metaeuk30.smag30.a3m",
                msa_format_mode=6,
                db_load_mode=db_load_mode,
                additional_params=filter_param,
            )
            mmseqs.rmdb(work_base / "res_env_exp_realign_filter")
            mmseqs.rmdb(work_base / "res_env_exp_realign")
            mmseqs.rmdb(work_base / "res_env_exp")
            mmseqs.rmdb(work_base / "res_env")

        if use_templates and template_db is not None:
            tdb = template_db.with_name(f"{template_db.name}{suffix3}")
            template_m8_dbtype = work_base / (template_db.name + ".dbtype")
            if not template_m8_dbtype.exists():
                mmseqs.search(
                    work_base / "prof_res",
                    template_db,
                    work_base / "res_pdb",
                    work_base / "tmp2",
                    additional_search_param=[
                        "--db-load-mode",
                        str(db_load_mode),
                        "-a",
                        "-e",
                        "0.1",
                        *template_search_param,
                    ],
                )
                mmseqs.convertalis(
                    work_base / "prof_res",
                    tdb,
                    work_base / "res_pdb",
                    work_base / template_db.name,
                    db_load_mode=db_load_mode,
                )
                mmseqs.rmdb(work_base / "res_pdb")

        mmseqs.mergedbs(qdb, work_base / "final.a3m", *used_dbs_for_merge)
        mmseqs.unpackdb(work_base / "final.a3m", unpack_dest)


def mmseqs_search_pair(
    mmseqs: MMseqs,
    base: Path,
    uniref_db: Path,
    spire_db: Path,
    *,
    pair_env: bool = True,
    prefilter_mode: int = 0,
    s: float | None = 8,
    db_load_mode: int = 2,
    pairing_strategy: int = 0,
    unpack: bool = True,
) -> None:
    """Run MMseqs2 paired MSA search (complex). Uses spire_db if pair_env else uniref_db."""
    db = spire_db if pair_env else uniref_db
    output_suffix = ".env.paired.a3m" if pair_env else ".paired.a3m"

    if not uniref_db.with_name(f"{uniref_db.name}.dbtype").is_file():
        raise FileNotFoundError(f"Database {uniref_db} does not exist")
    if (
        not uniref_db.with_name(f"{uniref_db.name}.idx").is_file()
        and not uniref_db.with_name(f"{uniref_db.name}.idx.index").is_file()
    ) or (os.environ.get("MMSEQS_IGNORE_INDEX") or "").lower() in (
        "1",
        "true",
        "yes",
    ):
        _logger.info("Search does not use index")
        db_load_mode = 0
        suffix1 = "_seq"
        suffix2 = "_aln"
    else:
        suffix1 = ".idx"
        suffix2 = ".idx"

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
    expand_param = [
        "--expansion-mode",
        "0",
        "-e",
        "inf",
        "--expand-filter-clusters",
        "0",
        "--max-seq-id",
        "0.95",
    ]

    qdb = base / "qdb"
    udb1 = db.with_name(f"{db.name}{suffix1}")
    udb2 = db.with_name(f"{db.name}{suffix2}")

    mmseqs.search(
        qdb,
        db,
        base / "res",
        base / "tmp",
        additional_search_param=search_param,
    )
    mmseqs.mvdb(base / "tmp/latest/profile_1", base / "prof_res")
    mmseqs.lndb(base / "qdb_h", base / "prof_res_h")
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
        align_eval=1,
        max_accept=1000000,
    )
    mmseqs.pairaln(
        qdb,
        db,
        base / "res_exp_realign",
        base / "res_exp_realign_pair",
        db_load_mode=db_load_mode,
        pairing_mode=pairing_strategy,
        pairing_dummy_mode=0,
    )
    mmseqs.align(
        base / "prof_res",
        udb1,
        base / "res_exp_realign_pair",
        base / "res_exp_realign_pair_bt",
        db_load_mode=db_load_mode,
        align_eval=10,
        max_accept=1000000,
    )
    mmseqs.pairaln(
        qdb,
        db,
        base / "res_exp_realign_pair_bt",
        base / "res_final",
        db_load_mode=db_load_mode,
        pairing_mode=pairing_strategy,
        pairing_dummy_mode=1,
    )
    mmseqs.result2msa(
        qdb,
        udb1,
        base / "res_final",
        base / "pair.a3m",
        msa_format_mode=5,
        db_load_mode=db_load_mode,
    )
    if unpack:
        mmseqs.unpackdb(
            base / "pair.a3m",
            base,
            unpack_suffix=output_suffix,
        )
    mmseqs.rmdb(base / "pair.a3m")
    mmseqs.rmdb(base / "res")
    mmseqs.rmdb(base / "res_exp")
    mmseqs.rmdb(base / "res_exp_realign")
    mmseqs.rmdb(base / "res_exp_realign_pair")
    mmseqs.rmdb(base / "res_exp_realign_pair_bt")
    mmseqs.rmdb(base / "res_final")
    mmseqs.rmdb(base / "prof_res")
    mmseqs.rmdb(base / "prof_res_h")


def run_search_from_path(
    query_path: Path,
    dbbase: Path,
    output_dir: Path,
    uniref_db: Path | None = None,
    metagenomic_db: Path | None = None,
    spire_db: Path | None = None,
    *,
    threads: int = 1,
    mmseqs: str | Path = "mmseqs",
    limit: bool = True,
    template_db: Path | None = None,
    use_templates: bool = False,
    use_env: bool = True,
    use_env_pairing: bool = False,
    pair_mode: Literal[
        "unpaired", "paired", "unpaired_paired"
    ] = "unpaired_paired",
    **kwargs: object,
) -> None:
    """Run MSA search from a FASTA/CSV/dir path. Results written as output_dir/<jobname>.a3m.

    DB paths are under dbbase if not absolute. For complex (multi-chain) input,
    runs monomer unpaired + optional paired search and merges per job.
    """
    if uniref_db is None:
        uniref_db = dbbase / "uniref30_2302_db"
    elif not uniref_db.is_absolute():
        uniref_db = dbbase / uniref_db
    if metagenomic_db is None:
        metagenomic_db = dbbase / "colabfold_envdb_202108_db"
    elif not metagenomic_db.is_absolute():
        metagenomic_db = dbbase / metagenomic_db
    if spire_db is None:
        spire_db = dbbase / "spire_ctg10_2401_db"
    elif not spire_db.is_absolute():
        spire_db = dbbase / spire_db

    queries, is_complex = get_queries_from_path(query_path)
    if not queries:
        raise ValueError(f"No queries found in {query_path}")

    # Build queries_unique: (jobname, query_seqs_unique, query_seqs_cardinality, other)
    queries_unique: list[list] = []
    for _job_number, (
        raw_jobname,
        query_sequences,
        _a3m_lines,
        other_molecules,
    ) in enumerate(queries):
        qs = (
            [query_sequences]
            if isinstance(query_sequences, str)
            else query_sequences
        )
        query_seqs_unique: list[str] = []
        for x in qs:
            if x not in query_seqs_unique:
                query_seqs_unique.append(x)
        query_seqs_cardinality = [0] * len(query_seqs_unique)
        for seq in qs:
            query_seqs_cardinality[query_seqs_unique.index(seq)] += 1
        queries_unique.append(
            [
                raw_jobname,
                query_seqs_unique,
                query_seqs_cardinality,
                other_molecules,
            ]
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    query_fas = output_dir / "query.fas"
    seq_id = 0
    with query_fas.open("w") as f:
        for _, query_sequences, _card, _ in queries_unique:
            seqs = (
                [query_sequences]
                if isinstance(query_sequences, str)
                else query_sequences
            )
            for seq in seqs:
                f.write(f">{101 + seq_id}\n{seq}\n")
                seq_id += 1

    runner = MMseqs(threads=threads, mmseqs=mmseqs)
    runner.createdb(
        query_fas,
        output_dir / "qdb",
        shuffle=0,
        dbtype=1,
    )
    with (output_dir / "qdb.lookup").open("w") as f:
        id_ = 0
        file_number = 0
        for raw_jobname, query_sequences, _, _ in queries_unique:
            seqs = (
                [query_sequences]
                if isinstance(query_sequences, str)
                else query_sequences
            )
            for _seq in seqs:
                f.write(f"{id_}\t{raw_jobname.split()[0]}\t{file_number}\n")
                id_ += 1
            file_number += 1

    metagenomic_db_actual = metagenomic_db if use_env else None
    run_monomer = not (is_complex and pair_mode == "paired")
    if run_monomer:
        mmseqs_search_monomer(
            runner,
            output_dir / "qdb",
            output_dir,
            uniref_db,
            metagenomic_db_actual,
            limit=limit,
            base=output_dir,
            template_db=template_db,
            use_templates=use_templates,
            **kwargs,
        )

    if is_complex:
        if pair_mode != "unpaired":
            mmseqs_search_pair(
                runner,
                output_dir,
                uniref_db,
                uniref_db,
                pair_env=False,
                prefilter_mode=kwargs.get("prefilter_mode", 0),
                s=kwargs.get("s"),
                db_load_mode=kwargs.get("db_load_mode", 2),
                pairing_strategy=kwargs.get("pairing_strategy", 0),
                unpack=True,
            )
        if pair_mode != "unpaired" and use_env_pairing:
            mmseqs_search_pair(
                runner,
                output_dir,
                uniref_db,
                spire_db,
                pair_env=True,
                prefilter_mode=kwargs.get("prefilter_mode", 0),
                s=kwargs.get("s"),
                db_load_mode=kwargs.get("db_load_mode", 2),
                pairing_strategy=kwargs.get("pairing_strategy", 0),
                unpack=True,
            )

        # Merge per-job: read unpaired/paired a3m by index, msa_to_str, write job_index.a3m
        id_ = 0
        for job_number, (
            _raw_jobname,
            query_sequences,
            query_seqs_cardinality,
            _,
        ) in enumerate(queries_unique):
            unpaired_msa: list[str] = []
            paired_msa: list[str] | None = (
                [] if len(query_seqs_cardinality) > 1 else None
            )
            seqs = (
                [query_sequences]
                if isinstance(query_sequences, str)
                else query_sequences
            )
            for _seq in seqs:
                if pair_mode != "paired":
                    a3m_path = output_dir / f"{id_}.a3m"
                    if a3m_path.exists():
                        unpaired_msa.append(a3m_path.read_text())
                        a3m_path.unlink()
                if pair_mode != "unpaired" and use_env_pairing:
                    env_paired = output_dir / f"{id_}.env.paired.a3m"
                    if env_paired.exists():
                        with (output_dir / f"{id_}.paired.a3m").open(
                            "a"
                        ) as fp:
                            fp.write(env_paired.read_text())
                        env_paired.unlink()
                if len(query_seqs_cardinality) > 1 and pair_mode != "unpaired":
                    paired_path = output_dir / f"{id_}.paired.a3m"
                    if paired_path.exists() and paired_msa is not None:
                        paired_msa.append(paired_path.read_text())
                        paired_path.unlink()
                id_ += 1
            out_unpaired = (
                None
                if pair_mode == "paired"
                else (unpaired_msa if unpaired_msa else None)
            )
            out_paired = None if pair_mode == "unpaired" else paired_msa
            msa = msa_to_str(
                out_unpaired, out_paired, seqs, query_seqs_cardinality
            )
            (output_dir / f"{job_number}.a3m").write_text(msa)

    # Rename to jobname.a3m and cleanup
    for job_number, (jobname, _, _, _) in enumerate(queries_unique):
        src = output_dir / f"{job_number}.a3m"
        if src.exists():
            src.rename(output_dir / f"{safe_filename(jobname)}.a3m")
    runner.rmdb(output_dir / "qdb")
    if (output_dir / "qdb_h.dbtype").exists():
        runner.rmdb(output_dir / "qdb_h")
    query_fas.unlink(missing_ok=True)
