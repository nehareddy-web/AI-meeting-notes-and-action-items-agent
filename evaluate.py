import re, unicodedata
import numpy as np
from scipy.optimize import linear_sum_assignment

def _tokens(x):
    s = unicodedata.normalize("NFKC", str(x)).lower()
    return set(re.findall(r"\w+", s))      # hyphens, unicode dashes, punctuation all split words

def leaf_score(p, g):
    if isinstance(g, (int, float, bool)) or g is None:
        return 1.0 if p == g else 0.0
    tp, tg = _tokens(p), _tokens(g)
    if not tp or not tg:
        return float(tp == tg)
    inter = len(tp & tg)
    if inter == 0:
        return 0.0
    pr, rc = inter / len(tp), inter / len(tg)
    return 2 * pr * rc / (pr + rc)

def score(p, g):
    if isinstance(g, dict):
        if not isinstance(p, dict):
            return 0.0
        return sum(score(p.get(k), v) for k, v in g.items()) / max(len(g), 1)
    if isinstance(g, list):
        if not isinstance(p, list) or not g:
            return float(p == g)
        if not p:
            return 0.0
        m = [[score(pj, gi) for pj in p] for gi in g]          # gold x predicted score matrix
        rows, cols = linear_sum_assignment(-np.array(m))       # best one-to-one matching (order independent)
        return float(sum(m[i][j] for i, j in zip(rows, cols))) / max(len(g), len(p))
    return leaf_score(p, g)

def top_level_scores(p, g):
    if not isinstance(g, dict):
        return {}
    p = p if isinstance(p, dict) else {}
    return {k: round(score(p.get(k), v), 3) for k, v in g.items()}
