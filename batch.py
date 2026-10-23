import json, pandas as pd
from concurrent.futures import ThreadPoolExecutor
import agent, db
from loader import load

def run_batch(modality, n=10, workers=4, offset=0, **options):
    graph = agent.build()
    recs = load(modality, limit=n, offset=offset)
    def one(r):
        try:
            return agent.run_record(graph, r, options)
        except Exception as e:
            print("FAILED", r["record_id"][:8], repr(e)[:150])
            return None
    with ThreadPoolExecutor(workers) as ex:
        outs = [o for o in ex.map(one, recs) if o]
    k = max(len(outs), 1)
    summary = {"experiment": options.get("experiment", "baseline"), "modality": modality, "done": len(outs),
               "failed": len(recs) - len(outs),
               "avg_score": round(sum(o["score"] or 0 for o in outs) / k, 3),
               "valid_rate": round(sum(o["valid"] for o in outs) / k, 2),
               "avg_attempts": round(sum(o["attempts"] for o in outs) / k, 2),
               "avg_in_tok": int(sum(o["in_tok"] for o in outs) / k),
               "avg_out_tok": int(sum(o["out_tok"] for o in outs) / k)}
    print("\nSUMMARY:", summary)
    return summary

def compare():
    return pd.read_sql("""SELECT experiment, modality, COUNT(*) AS n, ROUND(AVG(score),3) AS avg_score,
        ROUND(AVG(schema_valid),2) AS valid, ROUND(AVG(attempts),2) AS attempts,
        CAST(AVG(input_tokens) AS INT) AS in_tok, CAST(AVG(output_tokens) AS INT) AS out_tok
        FROM runs GROUP BY experiment, modality ORDER BY modality, avg_score DESC""", db.conn())

def weak_fields(experiment="baseline", modality="audio", min_count=2):
    rows = db.conn().execute("SELECT field_scores FROM runs WHERE experiment=? AND modality=?",
                             (experiment, modality)).fetchall()
    agg = {}
    for (fs,) in rows:
        for k, v in json.loads(fs or "{}").items():
            agg.setdefault(k, []).append(v)
    df = pd.DataFrame([{"field": k, "count": len(v), "avg_score": round(sum(v) / len(v), 3)}
                       for k, v in agg.items() if len(v) >= min_count])
    return df.sort_values("avg_score").head(15) if len(df) else df
