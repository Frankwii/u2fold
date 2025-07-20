from pathlib import Path
import subprocess
import os
import orjson
from u2fold.model.spec import U2FoldSpec

DOC_SCHEMA_FILE = Path("docs/spec/schema.json")

def dump_model_schema(target_file = DOC_SCHEMA_FILE):
    with open(target_file, 'w') as f:
        
        f.write(
            orjson.dumps(U2FoldSpec.model_json_schema()).decode()
        )

def generate_html(schema_file = DOC_SCHEMA_FILE):
    subprocess.run(
        ["generate-schema-doc", schema_file.name],
        cwd=schema_file.parent
    )

def show_html(html_dir = DOC_SCHEMA_FILE.parent):
    subprocess.Popen([
        os.getenv("BROWSER", "firefox"), "schema_doc.html"
     ],   cwd = html_dir
    )

if __name__ == "__main__":
    dump_model_schema()
    generate_html()
    show_html()
