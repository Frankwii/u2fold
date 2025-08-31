"""Manages a small sqlite3 database that stores training results (not weights!)"""

import sqlite3
from u2fold.model.neural_network_spec import NeuralNetworkSpec
from u2fold.model.spec import U2FoldSpec
from u2fold.model.common_namespaces import EpochMetricData
from u2fold.utils.get_directories import get_project_home


db_connection = sqlite3.connect(get_project_home() / "results.db")

table_columns = {
    "spec": "TEXT",
    "loss": "REAL",
    "consistency_loss": "REAL",
    "color_loss": "REAL",
    "fidelity_loss": "REAL",
    "ground_truth_loss": "REAL",
    "dssim": "REAL",
    "psnr_minimizable": "REAL",
    "uciqe_minimizable": "REAL",
}

if db_connection.execute("""
    SELECT name
    FROM sqlite_master
    WHERE type='table' AND name='results'
 """).fetchone() is None:
    _ = db_connection.execute(f"""
        CREATE TABLE results (
            {",\n".join(f'{k} {v}' for k,v in table_columns.items())}
        )
    """)

def save_training_result(
    spec: U2FoldSpec[NeuralNetworkSpec], result: EpochMetricData
) -> None:
    """Persists the results into a sqlite database"""

    register = {
        "spec": "?", # need to escape this since it contains especial characters
        "loss": result.overall_loss,
        **{
            k.replace(" ", "_").lower(): v
            for k, v in (result.granular_loss | result.metrics).items()
        },
    }

    _ = db_connection.execute(f"""
        INSERT INTO results ({",".join(register.keys())})
        VALUES ({",".join(map(str, register.values()))})
    """, (spec.model_dump_json(indent=2),))
