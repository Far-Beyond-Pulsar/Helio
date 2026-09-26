"""Read-only analysis of the v2 2M benchmark DB, aligned to GPU source frames.

Usage: python scripts/analyze_two_million.py two_million_verified.sqlite
GPU queries use the DB's deduplicated, epoch-aware source-frame views. CPU
measurements are queried separately: do not weight them by readback delivery.
The static summary uses only the last sample_frames of the initial baseline.
"""
import json
import sqlite3
import sys
from pathlib import Path
from statistics import mean, median


def summary(values):
    values = sorted(values)
    if not values:
        return None
    return {
        "n": len(values), "mean_ms": round(mean(values), 4),
        "median_ms": round(median(values), 4),
        "p95_ms": round(values[min(len(values) - 1, int(len(values) * .95))], 4),
    }


def analyze(path):
    db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    run = dict(db.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 1").fetchone())
    run_id = run["id"]
    warmup_end = max(0, run["initial_frames"] - run["sample_frames"])
    rows = []
    for cp in db.execute("SELECT * FROM checkpoints WHERE run_id=? ORDER BY sequence", [run_id]):
        cutoff = warmup_end if cp["sequence"] == 0 else 0
        cpu = [r[0] for r in db.execute("""
            SELECT f.frame_ms FROM frames f JOIN frame_context x ON x.frame_id=f.id
            WHERE f.checkpoint_id=? AND x.is_sample=1 AND f.frame>?
        """, [cp["id"], cutoff])]
        gpu = [r[0] for r in db.execute("""
            SELECT total_gpu_ms FROM gpu_frame_samples
            WHERE checkpoint_id=? AND is_sample=1 AND source_frame>?
        """, [cp["id"], cutoff]) if r[0] is not None]
        rows.append({"dynamic_objects": cp["dynamic_objects"], "cadence": summary(cpu),
                     "fps": round(1000 / mean(cpu), 2) if cpu else None, "gpu": summary(gpu)})
    final_cp = db.execute("SELECT id FROM checkpoints WHERE run_id=? ORDER BY sequence DESC LIMIT 1", [run_id]).fetchone()[0]
    passes = [dict(r) for r in db.execute("""
        SELECT pass_name, count(*) AS samples, round(avg(gpu_ms),5) AS gpu_ms
        FROM gpu_pass_samples WHERE checkpoint_id=? AND is_sample=1
        GROUP BY pass_name ORDER BY gpu_ms DESC
    """, [final_cp])]
    coverage = [r[0] for r in db.execute("""
        SELECT g.total_gpu_ms - sum(t.gpu_ms)
        FROM gpu_frame_samples g JOIN pass_timings t ON t.profiler_frame_id=g.profiler_frame_id
        WHERE g.checkpoint_id=? AND g.is_sample=1 AND substr(t.pass_name,1,2) != '__'
        GROUP BY g.profiler_frame_id
    """, [final_cp])]
    cpu_passes = [dict(r) for r in db.execute("""
        SELECT t.pass_name, round(avg(t.cpu_ms),5) AS cpu_ms FROM pass_timings t
        JOIN profiler_frames p ON p.id=t.profiler_frame_id
        JOIN frames f ON f.id=p.frame_id JOIN frame_context x ON x.frame_id=f.id
        WHERE f.checkpoint_id=? AND x.is_sample=1
        GROUP BY t.pass_name ORDER BY cpu_ms DESC
    """, [final_cp])]
    health = dict(db.execute("""
        SELECT max(p.readback_drops) AS readback_drops,max(p.query_overflows) AS query_overflows,
          max(p.gpu_lag_frames) AS max_gpu_lag_frames
        FROM profiler_frames p JOIN frames f ON f.id=p.frame_id WHERE f.run_id=?
    """, [run_id]).fetchone())
    metadata = dict(db.execute("SELECT key,value FROM run_metadata WHERE run_id=?", [run_id]))
    return {"database": str(Path(path).resolve()), "run": run, "metadata": metadata,
            "health": health, "checkpoints": rows, "final_gpu_passes": passes,
            "final_cpu_passes": cpu_passes, "graph_minus_pass_sum_ms": summary(coverage)}


if __name__ == "__main__":
    print(json.dumps(analyze(sys.argv[1]), indent=2))
