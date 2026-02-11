# gmol-core

Core components of GalaxyMol.

## Notes to developers

Before starting development, make sure to create the virtual environment and
install the pre-commit hooks by running the following command:

```bash
uv sync
uv run pre-commit install --install-hooks --overwrite
```

Before you push to the development branch, make sure to run mypy checks for
any type-related errors.

```bash
mypy [--pretty]
```

Although it's not a strict requirement, consider following the
[conventional commits](https://www.conventionalcommits.org/en/v1.0.0/)
guidelines for commit messages. This will help reviewers to understand the
changes in a more structured way.

## License

This project is licensed under the Apache License, Version 2.0. See the
[LICENSE](LICENSE) file for details.
