# File structure of this project
## Top-level folders
There are three main folders under the root of the project: `doc`, `src` and `test`.

`docs` contains all of the "human-readable" part of the project: Markdown, LaTeX and PDF files, including this very file.

`src` contains all of the "package" code. It will likely be written entirely in Python as most popular ML tools are available on it. Its module (code) structure is explained [here](module_structure.md).

`tests` contains all of the tests for the "package" code.

The rest of the folders are there mainly for development purposes and are subject to modification or removal.

## pyproject.toml
The file [pyproject.toml] is important since it holds dependency and build-related information, as specified in [PEP 621](https://peps.python.org/pep-0621/).
