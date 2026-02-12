"""
Some of the code is imported from DeepMind's AlphaFold.
See https://github.com/google-deepmind/alphafold/blob/main/alphafold/data/tools/hhsearch.py
for the original code.

.. admonition:: License
    :collapsible: closed

    .. code-block::

        Copyright 2021 DeepMind Technologies Limited

        Licensed under the Apache License, Version 2.0 (the "License");
        you may not use this file except in compliance with the License.
        You may obtain a copy of the License at

            http://www.apache.org/licenses/LICENSE-2.0

        Unless required by applicable law or agreed to in writing, software
        distributed under the License is distributed on an "AS IS" BASIS,
        WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        See the License for the specific language governing permissions and
        limitations under the License.
"""

import logging
import subprocess as sp
from pathlib import Path
from typing import ClassVar

_logger = logging.getLogger(__name__)


class HHSearch:
    """Python wrapper of the HHsearch binary."""

    output_format: ClassVar[str] = "hhr"
    input_format: ClassVar[str] = "a3m"

    def __init__(
        self,
        *,
        binary_path: Path,
        databases: list[Path],
        output_dir: Path,
        maxseq: int = 1_000_000,
    ):
        """Initializes the Python HHsearch wrapper.

        Args:
          binary_path: The path to the HHsearch executable.
          databases: A sequence of HHsearch database paths. This should be the
            common prefix for the database files (i.e. up to but not including
            _hhm.ffindex etc.)
          maxseq: The maximum number of rows in an input alignment. Note that this
            parameter is only supported in HHBlits version 3.1 and higher.
        """
        self.binary_path = binary_path
        self.databases = databases
        self.maxseq = maxseq
        self.output_dir = output_dir

        for database_path in self.databases:
            if not list(database_path.parent.glob(f"{database_path.name}_*")):
                raise ValueError(
                    f"Could not find HHsearch database {database_path}"
                )

        self.output_dir.mkdir(exist_ok=True, parents=True)

    def query(self, a3m: Path):
        """Queries the database using HHsearch using a given a3m."""
        db_cmd: list[str | Path] = []
        for db in self.databases:
            db_cmd.append("-d")
            db_cmd.append(db)

        # fmt: off
        cmd: list[str | Path] = [
            self.binary_path,
            "-i", a3m,
            "-o", self.output_dir / "output.hhr",
            "-maxseq", str(self.maxseq),
            *db_cmd,
        ]
        # fmt: on

        _logger.info('Launching subprocess "%s"', cmd)
        sp.run(cmd, check=True)
