import os, requests, jsonschema
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
import llm, db, preprocess
from evaluate import score as score_fn, top_level_scores

MAX_ATTEMPTS = 3

class State(TypedDict):
    record: dict
    options: dict
    context: str
    stats: dict
    prediction: Optional[dict]
    feedback: Optional[str]
    attempts: int
    valid: bool
    grounded: Optional[bool]
    score: Optional[float]
    field_scores: Optional[dict]
    in_tok: int
    out_tok: int
    rejected: bool
    run_id: Optional[int]

# ---------- nodes ----------
def prep(state: State):
    r = state["record"]
    if state["options"].get("preprocess", True):
        ctx, stats = preprocess.run(r["modality"], r["context"])
    else:
        ctx, stats = r["context"], {}
    return {"context": ctx, "stats": stats}

def extract(state: State):
    pred, u = llm.call_extract(state["context"], state["record"]["question"],
                               state["record"]["json_schema"], state["feedback"])
    return {"prediction": pred, "attempts": state["attempts"] + 1,
            "in_tok": state["in_tok"] + u[0], "out_tok": state["out_tok"] + u[1]}

def validate(state: State):
    if state["prediction"] is None:
        return {"valid": False, "feedback": "No tool output returned (maybe truncated). Give a complete, concise answer."}
    try:
        jsonschema.validate(state["prediction"], state["record"]["json_schema"])
        return {"valid": True, "feedback": None}
    except jsonschema.ValidationError as e:
        path = "/".join(map(str, e.absolute_path))
        return {"valid": False, "feedback": f"Schema error at '{path}': {e.message[:250]}"}

def after_validate(state: State):
    if not state["valid"]:
        return "extract" if state["attempts"] < MAX_ATTEMPTS else "grade"
    return "critique" if state["options"].get("reflect") else "grade"

def critique(state: State):
    v, u = llm.call_critic(state["context"], state["record"]["question"], state["prediction"])
    out = {"grounded": v["grounded"], "in_tok": state["in_tok"] + u[0], "out_tok": state["out_tok"] + u[1]}
    if not v["grounded"]:
        out["feedback"] = "These parts are not supported by the context: " + "; ".join(v["issues"])
    return out

def after_critique(state: State):
    return "extract" if (not state["grounded"] and state["attempts"] < MAX_ATTEMPTS) else "grade"

def grade(state: State):
    gt = state["record"].get("ground_truth")
    if gt is None:
        return {"score": None, "field_scores": None}
    return {"score": score_fn(state["prediction"], gt),
            "field_scores": top_level_scores(state["prediction"], gt)}

def approve(state: State):
    if not state["options"].get("approve"):
        return {}
    answer = interrupt({"question": "Save this result? (yes/no)", "prediction": state["prediction"]})
    return {"rejected": str(answer).strip().lower() not in ("y", "yes")}

def after_approve(state: State):
    return END if state["rejected"] else "save"

def save(state: State):
    rid = db.save_run(state["record"], state["prediction"], state["valid"], state["grounded"],
                      state["score"], state["field_scores"], state["attempts"],
                      state["in_tok"], state["out_tok"],
                      state["options"].get("experiment", "baseline"), state["stats"])
    return {"run_id": rid}

def notify(state: State):
    n = len(db.find_action_items(state["prediction"]))
    msg = (f"[{state['options'].get('experiment')}] run {state['run_id']}: {n} action items, "
           f"score={None if state['score'] is None else round(state['score'], 3)}, attempts={state['attempts']}")
    hook = os.getenv("SLACK_WEBHOOK_URL")
    if hook:
        requests.post(hook, json={"text": msg}, timeout=10)
    print(msg)
    return {}

# ---------- graph ----------
def build(checkpointer=None):
    db.init_db()
    g = StateGraph(State)
    for name, fn in [("prep", prep), ("extract", extract), ("validate", validate), ("critique", critique),
                     ("grade", grade), ("approve", approve), ("save", save), ("notify", notify)]:
        g.add_node(name, fn)
    g.add_edge(START, "prep")
    g.add_edge("prep", "extract")
    g.add_edge("extract", "validate")
    g.add_conditional_edges("validate", after_validate, ["extract", "critique", "grade"])
    g.add_conditional_edges("critique", after_critique, ["extract", "grade"])
    g.add_edge("grade", "approve")
    g.add_conditional_edges("approve", after_approve, ["save", END])
    g.add_edge("save", "notify")
    g.add_edge("notify", END)
    return g.compile(checkpointer=checkpointer)

DEFAULTS = {"experiment": "baseline", "preprocess": True, "reflect": False, "approve": False}

def run_record(graph, record, options=None, thread_id=None):
    init = {"record": record, "options": {**DEFAULTS, **(options or {})}, "context": "", "stats": {},
            "prediction": None, "feedback": None, "attempts": 0, "valid": False, "grounded": None,
            "score": None, "field_scores": None, "in_tok": 0, "out_tok": 0, "rejected": False, "run_id": None}
    cfg = {"configurable": {"thread_id": thread_id or record["record_id"]}}
    return graph.invoke(init, cfg)
