"""
Some of the code is imported from ColabFold.
See:

* https://github.com/sokrypton/ColabFold/blob/main/colabfold/batch.py
* https://github.com/sokrypton/ColabFold/blob/main/colabfold/input.py
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
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal

import pandas as pd

_logger = logging.getLogger(__name__)


@dataclass
class NonProteinQuery:
    class Type(Enum):
        RNA = ("sequence", "rna")
        DNA = ("sequence", "dna")
        CCD = ("ccdCodes", "ligand")
        SMILES = ("smiles", "ligand")

        @property
        def af3code(self):
            return self.value[0]

        @property
        def upperclass(self):
            return self.value[1]

    moltype: Type
    definition: str
    copies: int = 1


@dataclass
class MsaQuery:
    id: str
    seqs: list[str]
    prev_msa: str = ""
    non_proteins: list[NonProteinQuery] = field(default_factory=list)

    @classmethod
    def from_sequence(cls, seq_id: str, sequence: str) -> "MsaQuery":
        parts = sequence.split(":")
        if len(parts) == 1:
            return cls(seq_id, [sequence.upper()])

        protein_queries, other = classify_molecules(sequence)
        return cls(seq_id, protein_queries, non_proteins=other)

    @property
    def is_complex(self) -> bool:
        if len(self.seqs) > 1 or len(self.non_proteins) > 0:
            return True

        if self.prev_msa and self.prev_msa.startswith("#"):
            tab_sep = self.prev_msa.splitlines()[0][1:].split("\t")
            if len(tab_sep) == 2:
                card = list(map(int, tab_sep[1].split(",")))
                if len(card) > 1 or (len(card) == 1 and card[0] > 1):
                    return True

        return False


def parse_fasta(fasta_string: str) -> tuple[list[str], list[str]]:
    """Parse FASTA string into (sequences, descriptions)."""
    sequences: list[str] = []
    descriptions: list[str] = []
    index = -1
    for line in fasta_string.splitlines():
        line = line.strip()
        if line.startswith("#"):
            continue
        if line.startswith(">"):
            index += 1
            descriptions.append(line[1:])
            sequences.append("")
            continue
        if not line:
            continue
        sequences[index] += line
    return sequences, descriptions


def classify_molecules(
    query_sequence: str,
) -> tuple[list[str], list[NonProteinQuery]]:
    """Split sequence line into protein parts and optional other mol types."""
    sequences = query_sequence.split(":")

    protein_queries: list[str] = []
    other_queries: list[NonProteinQuery] = []
    for seq in sequences:
        if seq.count("|") == 0:
            protein_queries.append(seq.upper())
        else:
            parts = seq.split("|")
            moltype, sequence, *rest = parts
            moltype = NonProteinQuery.Type[moltype.upper()]
            if moltype == NonProteinQuery.Type.SMILES:
                sequence = sequence.replace(";", ":")
            else:
                sequence = sequence.upper()
            copies = int(rest[0]) if rest else 1
            other_queries.append(NonProteinQuery(moltype, sequence, copies))

    return protein_queries, other_queries


def pair_sequences(
    a3m_lines: list[str],
    query_sequences: list[str],
    query_cardinality: list[int],
) -> str:
    """Combine per-chain A3M into one paired A3M block."""
    a3m_line_paired = [""] * len(a3m_lines[0].splitlines())
    for n, _seq in enumerate(query_sequences):
        lines = a3m_lines[n].splitlines()
        for i, line in enumerate(lines):
            if line.startswith(">"):
                if n != 0:
                    line = line.replace(">", "\t", 1)
                a3m_line_paired[i] = a3m_line_paired[i] + line
            else:
                a3m_line_paired[i] = (
                    a3m_line_paired[i] + line * query_cardinality[n]
                )
    return "\n".join(a3m_line_paired)


def pad_sequences(
    a3m_lines: list[str],
    query_sequences: list[str],
    query_cardinality: list[int],
) -> str:
    """Pad per-chain A3M into one unpaired (padded) A3M block."""
    _blank_seq = [
        "-" * len(seq)
        for n, seq in enumerate(query_sequences)
        for _ in range(query_cardinality[n])
    ]
    a3m_lines_combined: list[str] = []
    pos = 0
    for n, _seq in enumerate(query_sequences):
        for _j in range(query_cardinality[n]):
            lines = a3m_lines[n].split("\n")
            for a3m_line in lines:
                if len(a3m_line) == 0:
                    continue
                if a3m_line.startswith(">"):
                    a3m_lines_combined.append(a3m_line)
                else:
                    a3m_lines_combined.append(
                        "".join(
                            (
                                *_blank_seq[:pos],
                                a3m_line,
                                *_blank_seq[pos + 1 :],
                            )
                        )
                    )
            pos += 1
    return "\n".join(a3m_lines_combined)


def pair_msa(
    query_seqs_unique: list[str],
    query_seqs_cardinality: list[int],
    paired_msa: list[str] | None,
    unpaired_msa: list[str] | None,
) -> str:
    """Merge paired and/or unpaired MSA into one A3M string."""
    if paired_msa is None and unpaired_msa is not None:
        return pad_sequences(
            unpaired_msa, query_seqs_unique, query_seqs_cardinality
        )
    if paired_msa is not None and unpaired_msa is not None:
        return (
            pair_sequences(
                paired_msa, query_seqs_unique, query_seqs_cardinality
            )
            + "\n"
            + pad_sequences(
                unpaired_msa, query_seqs_unique, query_seqs_cardinality
            )
        )
    if paired_msa is not None and unpaired_msa is None:
        return pair_sequences(
            paired_msa, query_seqs_unique, query_seqs_cardinality
        )
    raise ValueError("Invalid pairing")


def msa_to_str(
    unpaired_msa: list[str] | None,
    paired_msa: list[str] | None,
    query_seqs_unique: list[str],
    query_seqs_cardinality: list[int],
) -> str:
    """Format paired/unpaired MSA as ColabFold-style A3M with # header."""
    msa = "#" + ",".join(map(str, map(len, query_seqs_unique))) + "\t"
    msa += ",".join(map(str, query_seqs_cardinality)) + "\n"
    query_seqs_cardinality = [1 for _ in query_seqs_cardinality]
    msa += pair_msa(
        query_seqs_unique, query_seqs_cardinality, paired_msa, unpaired_msa
    )
    return msa


def get_queries(
    input_path: Path,
    sort_queries_by: Literal["length", "random"] | None = "length",
) -> list[MsaQuery]:
    """Read FASTA, CSV/TSV, A3M or directory. Returns queries."""
    queries: list[MsaQuery] = []

    if input_path.is_file():
        if input_path.suffix in (".csv", ".tsv"):
            sep = "\t" if input_path.suffix == ".tsv" else ","
            df = pd.read_csv(input_path, sep=sep, dtype=str)
            queries.extend(
                MsaQuery.from_sequence(row.id, row.sequence)
                for row in df.itertuples(index=False)
            )
        elif input_path.suffix == ".a3m":
            seqs, _ = parse_fasta(input_path.read_text())
            if len(seqs) == 0:
                raise ValueError(f"{input_path} is empty")
            queries.append(
                MsaQuery(
                    input_path.stem,
                    [seqs[0]],
                    prev_msa=input_path.read_text(),
                )
            )
        elif input_path.suffix.lower() in (".fasta", ".faa", ".fa"):
            sequences, headers = parse_fasta(input_path.read_text())
            queries.extend(
                MsaQuery.from_sequence(header, sequence)
                for sequence, header in zip(sequences, headers)
            )
        else:
            raise ValueError(f"Unknown file format {input_path.suffix}")
    else:
        for file in sorted(input_path.iterdir()):
            if not file.is_file():
                continue

            kind = file.suffix.lower()
            if kind not in (".a3m", ".fasta", ".faa", ".fa"):
                _logger.warning("Skipping unsupported file: %s", file)
                continue

            content = file.read_text()
            seqs, _ = parse_fasta(content)
            if len(seqs) == 0:
                _logger.error("%s is empty", file)
                continue

            seq = seqs[0]
            if len(seqs) > 1 and kind != ".a3m":
                _logger.warning(
                    (
                        "More than one sequence in %s, ignoring all but the "
                        "first sequence"
                    ),
                    file,
                )

            if kind == ".a3m":
                queries.append(
                    MsaQuery(file.stem, [seq.upper()], prev_msa=content)
                )
            else:
                queries.append(MsaQuery.from_sequence(file.stem, seq))

    if sort_queries_by == "length":
        queries.sort(key=lambda t: len("".join(t.seqs)))
    elif sort_queries_by == "random":
        random.shuffle(queries)

    return queries
