"""
Some of the code is imported from alphafold3-pytorch.
See https://github.com/lucidrains/alphafold3-pytorch/blob/main/scripts/cluster_pdb_train_mmcifs.py.

.. admonition:: License
    :collapsible: closed

    .. code-block::

        MIT License

        Copyright (c) 2024 Phil Wang

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
import subprocess as sp
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

_logger = logging.getLogger(__name__)


def cluster_sequences_using_mmseqs2(
    input_filepath: Path,
    output_filepath: Path,
    min_seq_id: float = 0.5,
    coverage: float = 0.8,
    coverage_mode: Literal[0, 1, 2, 3] = 1,
    extra_parameters: dict[str, int | float | str] | None = None,
):
    """
    Run MMseqs2 on the input FASTA file and write the resulting clusters to
    a local output directory.
    """
    if not input_filepath.is_file():
        raise FileNotFoundError(f"Input file {input_filepath} not found.")
    if extra_parameters is None:
        extra_parameters = {}

    output_filepath.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory() as tmp_output_dir:
        mmseqs_command: list[str | Path] = [
            "mmseqs",
            "easy-cluster",
            input_filepath,
            output_filepath,
            tmp_output_dir,
            "--min-seq-id",
            str(min_seq_id),
            "-c",
            str(coverage),
            "--cov-mode",
            str(coverage_mode),
        ]
        for key, value in extra_parameters.items():
            mmseqs_command.append(key)
            mmseqs_command.append(str(value))

        _logger.info("Running MMseqs2 with command: %s", mmseqs_command)
        sp.run(mmseqs_command, check=False)
