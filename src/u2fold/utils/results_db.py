"""Manages a small sqlite3 database that stores training results (not weights!)"""

import sqlite3
from u2fold.model.neural_network_spec import NeuralNetworkSpec
from u2fold.model.spec import U2FoldSpec
from u2fold.model.common_namespaces import EpochMetricData
from u2fold.utils.get_directories import get_project_home


DB_PATH = get_project_home() / "results.db"
table_columns = {
    "spec": "TEXT",
    "loss": "REAL",
    "consistency_loss": "REAL",
    "color_cosine_similarity_loss": "REAL",
    "fidelity_loss": "REAL",
    "ground_truth_loss": "REAL",
    "dssim": "REAL",
    "psnr_minimizable": "REAL",
    "uciqe_minimizable": "REAL",
}

with sqlite3.connect(DB_PATH) as db_connection:
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
            f'{k.replace(" ", "_").lower()}_loss': v
            for k, v in result.granular_loss.items()
        },
        **{
            k.replace(" ", "_").lower():v
            for k, v in result.metrics.items()
        }
    }

    with sqlite3.connect(DB_PATH) as db_connection:
        _ = db_connection.execute(f"""
            INSERT INTO results ({",".join(register.keys())})
            VALUES ({",".join(map(str, register.values()))})
        """, (spec.model_dump_json(indent=2),))

def spec_is_in_db(spec: U2FoldSpec[NeuralNetworkSpec]) -> bool:
    with sqlite3.connect(DB_PATH) as db_connection:
        result = db_connection.execute("""
            SELECT * FROM results
            WHERE spec=?
        """, (spec.model_dump_json(indent=2),)).fetchone()

    return result is not None

def get_results_from_spec(spec: U2FoldSpec[NeuralNetworkSpec]) -> dict[str, float]:

    metrics = list(set(table_columns.keys()) - {"spec"})
    with sqlite3.connect(DB_PATH) as db_connection:
        result = db_connection.execute(f"""
            SELECT {','.join(metrics)}
            FROM results
            WHERE spec=?
        """, (spec.model_dump_json(indent=2),)).fetchone()

    return dict(zip(metrics, result))
