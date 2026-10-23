import sqlite3, json, re

def conn():
    return sqlite3.connect("meetings.db", timeout=30)

def init_db():
    c = conn()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS runs(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment TEXT, record_id TEXT, modality TEXT, question TEXT,
        prediction TEXT, schema_valid INTEGER, grounded INTEGER,
        score REAL, field_scores TEXT, attempts INTEGER,
        input_tokens INTEGER, output_tokens INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
    CREATE TABLE IF NOT EXISTS action_items(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER, assignee TEXT, task TEXT, due TEXT,
        status TEXT DEFAULT 'pending',
        FOREIGN KEY(run_id) REFERENCES runs(id));
    CREATE TABLE IF NOT EXISTS speaker_stats(
        run_id INTEGER, speaker TEXT, words INTEGER, share REAL,
        FOREIGN KEY(run_id) REFERENCES runs(id));
    """)
    c.commit(); c.close()

_A = re.compile(r"assign|owner|responsib")          # who
_NOT_ROLE = re.compile(r"role")
_DUE = re.compile(r"due|deadline|timeline")           # when
_TASK_PRIORITY = [re.compile(p) for p in (r"task", r"action", r"description", r"next_step")]

def _first(d, rx, exclude=None):
    for k, v in d.items():
        kl = k.lower()
        if rx.search(kl) and not (exclude and exclude.search(kl)) and isinstance(v, (str, int, float)) and v not in ("", None):
            return str(v)
    return None

def _task(d):
    for rx in _TASK_PRIORITY:
        for k, v in d.items():
            kl = k.lower()
            if rx.search(kl) and not _A.search(kl) and not _DUE.search(kl) and isinstance(v, str) and v:
                return v
    return None

def find_action_items(obj):
    # any list of dicts that has an assignee-like key AND a task-like key counts as action items
    out = []
    if isinstance(obj, dict):
        for v in obj.values():
            out += find_action_items(v)
    elif isinstance(obj, list):
        for it in obj:
            if isinstance(it, dict):
                a = _first(it, _A, _NOT_ROLE)
                t = _task(it)
                if a and t:
                    out.append((a, t, _first(it, _DUE)))
                    continue
            out += find_action_items(it)
    return out

def save_run(record, pred, valid, grounded, score, field_scores, attempts,
             in_tok, out_tok, experiment, stats):
    c = conn(); cur = c.cursor()
    cur.execute("""INSERT INTO runs(experiment,record_id,modality,question,prediction,schema_valid,
                   grounded,score,field_scores,attempts,input_tokens,output_tokens)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (experiment, record["record_id"], record["modality"], record["question"],
                 json.dumps(pred), int(valid), None if grounded is None else int(grounded),
                 score, json.dumps(field_scores), attempts, in_tok, out_tok))
    rid = cur.lastrowid
    for a, t, d in find_action_items(pred):
        cur.execute("INSERT INTO action_items(run_id,assignee,task,due) VALUES (?,?,?,?)", (rid, a, t, d))
    for spk, s in (stats or {}).items():
        cur.execute("INSERT INTO speaker_stats VALUES (?,?,?,?)", (rid, spk, s["words"], s["share"]))
    c.commit(); c.close()
    return rid
