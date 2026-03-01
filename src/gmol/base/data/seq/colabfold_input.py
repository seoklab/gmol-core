"""
CLI-oriented input utilities for ColabFold-style MMseqs search.
FASTA, CSV/TSV, A3M, directory of FASTA; complex format A:B; MolType for non-protein.
"""

import logging
import random
from enum import Enum
from pathlib import Path

import pandas as pd

_logger = logging.getLogger(__name__)


class MolType(Enum):
    """Minimal mol types for sequence line parsing (e.g. SEQ:RNA|ACGU|1)."""

    RNA = ("sequence", "rna")
    DNA = ("sequence", "dna")
    CCD = ("ccdCodes", "ligand")
    SMILES = ("smiles", "ligand")

    def __init__(self, af3code: str, upperclass: str) -> None:
        self.af3code = af3code
        self.upperclass = upperclass

    @classmethod
    def get_moltype(cls, moltype: str) -> "MolType":
        if moltype == "RNA":
            return cls.RNA
        if moltype == "DNA":
            return cls.DNA
        if moltype == "SMILES":
            return cls.SMILES
        if moltype == "CCD":
            return cls.CCD
        raise ValueError(
            "Only dna, rna, ccd, smiles are allowed as molecule types."
        )


def safe_filename(file: str) -> str:
    """Return a filesystem-safe version of the string."""
    return "".join(
        c if c.isalnum() or c in ["_", ".", "-"] else "_" for c in file
    )


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
) -> tuple[list[str], list[tuple[MolType, str, int]] | None]:
    """Split sequence line into protein parts and optional other mol types."""
    sequences = query_sequence.upper().split(":")
    protein_queries: list[str] = []
    other_queries: list[tuple[MolType, str, int]] = []
    for seq in sequences:
        if seq.count("|") == 0:
            protein_queries.append(seq)
        else:
            parts = seq.split("|")
            moltype, sequence, *rest = parts
            moltype = MolType.get_moltype(moltype)
            if moltype == MolType.SMILES:
                sequence = sequence.replace(";", ":")
            copies = int(rest[0]) if rest else 1
            other_queries.append((moltype, sequence, copies))
    if len(other_queries) == 0:
        other_queries = []
    return protein_queries, other_queries if other_queries else None


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


QueryTuple = tuple[
    str,
    str | list[str],
    list[str] | None,
    list[tuple[MolType, str, int]] | None,
]


def get_queries_from_path(
    input_path: str | Path,
    sort_queries_by: str = "length",
) -> tuple[list[QueryTuple], bool]:
    """Read FASTA, CSV/TSV, A3M or directory. Returns (queries, is_complex)."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise OSError(f"{input_path} could not be found")

    queries: list[QueryTuple] = []

    if input_path.is_file():
        if input_path.suffix in (".csv", ".tsv"):
            sep = "\t" if input_path.suffix == ".tsv" else ","
            df = pd.read_csv(input_path, sep=sep, dtype=str)
            if "id" not in df.columns or "sequence" not in df.columns:
                raise ValueError(
                    "CSV/TSV must have 'id' and 'sequence' columns"
                )
            for row in df.itertuples(index=False):
                seq_id = str(row.id)
                sequence = str(row.sequence).upper()
                parts = sequence.split(":")
                if len(parts) == 1:
                    queries.append((seq_id, sequence, None, None))
                else:
                    protein_queries, other = classify_molecules(sequence)
                    queries.append((seq_id, protein_queries, None, other))
        elif input_path.suffix == ".a3m":
            seqs, _ = parse_fasta(input_path.read_text())
            if len(seqs) == 0:
                raise ValueError(f"{input_path} is empty")
            queries.append(
                (input_path.stem, seqs[0], [input_path.read_text()], None)
            )
        elif input_path.suffix.lower() in (".fasta", ".faa", ".fa"):
            sequences, headers = parse_fasta(input_path.read_text())
            for sequence, header in zip(sequences, headers):
                sequence = sequence.upper()
                if sequence.count(":") == 0:
                    queries.append((header, sequence, None, None))
                else:
                    protein_queries, other = classify_molecules(sequence)
                    queries.append((header, protein_queries, None, other))
        elif input_path.suffix.lower() in (".pdb", ".cif"):
            raise ValueError("PDB/CIF not supported. Use FASTA or CSV input.")
        else:
            raise ValueError(f"Unknown file format {input_path.suffix}")
    else:
        if not input_path.is_dir():
            raise ValueError("Expected a file or directory")
        for file in sorted(input_path.iterdir()):
            if not file.is_file():
                continue
            if file.suffix.lower() not in (".a3m", ".fasta", ".faa", ".fa"):
                _logger.warning("Skipping unsupported file: %s", file)
                continue
            if file.suffix.lower() == ".a3m":
                content = file.read_text()
                seqs, _ = parse_fasta(content)
                if len(seqs) == 0:
                    _logger.error("%s is empty", file)
                    continue
                queries.append((file.stem, seqs[0].upper(), [content], None))
            else:
                seqs, _ = parse_fasta(file.read_text())
                if len(seqs) == 0:
                    _logger.error("%s is empty", file)
                    continue
                q = seqs[0].upper()
                if q.count(":") == 0:
                    queries.append((file.stem, q, None, None))
                else:
                    protein_queries, other = classify_molecules(q)
                    queries.append((file.stem, protein_queries, None, other))

    if sort_queries_by == "length":
        queries.sort(
            key=lambda t: len(
                "".join(t[1]) if isinstance(t[1], list) else t[1]
            )
        )
    elif sort_queries_by == "random":
        random.shuffle(queries)

    is_complex = False
    for _, query_sequence, a3m_lines, _ in queries:
        if isinstance(query_sequence, list):
            is_complex = True
            break
        if a3m_lines is not None and a3m_lines[0].startswith("#"):
            tab_sep = a3m_lines[0].splitlines()[0][1:].split("\t")
            if len(tab_sep) == 2:
                card = list(map(int, tab_sep[1].split(",")))
                if len(card) > 1 or (len(card) == 1 and card[0] > 1):
                    is_complex = True
                    break
    return queries, is_complex
