from pathlib import Path
import subprocess
import os
import orjson
from u2fold.model.spec import U2FoldSpec
from u2fold.utils.get_directories import get_project_home
import sys

PYTHON_VENV_BINARIES = Path(sys.executable).parent
SPEC_DOCS_DIR = get_project_home() / Path("docs/spec")

def _dump_model_schema(target_file: Path):
    with open(target_file, 'w') as f:
        
        f.write(
            orjson.dumps(U2FoldSpec.model_json_schema()).decode()
        )

def _generate_html(schema_file: Path):
    args = [PYTHON_VENV_BINARIES / "generate-schema-doc", schema_file.name]
    subprocess.run(
        args,
        cwd=schema_file.parent,
        stdout=subprocess.DEVNULL
    )

def _show_html(html_dir: Path):
    subprocess.Popen([
        os.getenv("BROWSER", "firefox"), "schema_doc.html"
     ],   cwd = html_dir
    )

def show_docs():
    html_dir = SPEC_DOCS_DIR
    schema_file = SPEC_DOCS_DIR / "schema.json"
    html_dir.mkdir(parents=True, exist_ok=True)


    _dump_model_schema(schema_file)
    _generate_html(schema_file)
    _show_html(html_dir)

if __name__ == "__main__":
    show_docs()
