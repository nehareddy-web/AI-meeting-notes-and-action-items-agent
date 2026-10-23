import os, json, time, hashlib, threading, collections

MODEL = "openai/gpt-oss-120b"     # check console.groq.com/docs/models for current names
TPM = 7000                        # free tier ~8000 tokens/min. Paid tier: raise this (e.g. 100000)
MAX_DIRECT_CHARS = 18000          # longer contexts are condensed chunk-by-chunk first
CHUNK_CHARS = 14000

SYSTEM = ("You extract structured data from the given context. Use ONLY facts stated in the context; "
          "never invent values. For meeting transcripts, speakers appear as IDs like MEO069 - use those IDs "
          "as assignees. The transcript is noisy ASR text, so interpret it sensibly. "
          "Keep every string value short: one concise sentence, copying the speakers' own wording where possible. "
          "For action items include only tasks a speaker explicitly commits to or is assigned; set the deadline to "
          "the time phrase actually said (e.g. 'Wednesday afternoon'), never a guess. "
          "When asked about tasks per person, include one entry for EVERY person who is assigned something, "
          "and never merge different people's tasks into one entry. "
          "Return ONLY a JSON object that matches the required schema.")

CRITIC_SCHEMA = {"type": "object",
                 "properties": {"grounded": {"type": "boolean"},
                                "issues": {"type": "array", "items": {"type": "string"}}},
                 "required": ["grounded", "issues"]}

_client = None
_lock = threading.Lock()
_window = collections.deque()     # [timestamp, tokens] of calls in the last 60s
_notes_cache = {}

def client():
    global _client
    if _client is None:
        from groq import Groq
        _client = Groq(max_retries=5)      # reads GROQ_API_KEY from environment
    return _client

def _throttle(est):
    with _lock:
        while True:
            now = time.time()
            while _window and now - _window[0][0] > 60:
                _window.popleft()
            used = sum(t for _, t in _window)
            if used + est <= TPM or not _window:
                break
            wait = max(60 - (now - _window[0][0]) + 0.5, 1)
            print(f"   (rate limit: waiting {wait:.0f}s)")
            time.sleep(wait)
        entry = [time.time(), est]
        _window.append(entry)
        return entry

def _chat(system, user, max_out, response_format=None):
    est = (len(system) + len(user)) // 4 + min(max_out, 1500)
    entry = _throttle(est)
    kw = dict(model=MODEL, temperature=0, max_completion_tokens=max_out,
              messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    if MODEL.startswith("openai/gpt-oss"):
        kw["reasoning_effort"] = "low"
    if response_format:
        kw["response_format"] = response_format
    resp = client().chat.completions.create(**kw)
    u = (resp.usage.prompt_tokens, resp.usage.completion_tokens)
    entry[1] = u[0] + u[1]                  # replace estimate with real usage
    return resp.choices[0].message.content or "", u

def _parse(text):
    try:
        return json.loads(text)
    except Exception:
        a, b = text.find("{"), text.rfind("}")
        if a != -1 and b > a:
            try:
                return json.loads(text[a:b + 1])
            except Exception:
                return None
    return None

def _chat_json(system, user, schema, max_out):
    rf = {"type": "json_schema", "json_schema": {"name": "submit", "schema": schema}}
    try:
        text, u = _chat(system, user, max_out, rf)
    except Exception as e:
        if getattr(e, "status_code", None) != 400:
            raise
        # model/schema not supported by json_schema mode -> fall back to plain JSON mode
        user2 = user + "\n\nRequired JSON schema:\n" + json.dumps(schema)
        text, u = _chat(system, user2, max_out, {"type": "json_object"})
    return _parse(text), u

def _schema_hint(schema, prefix="", out=None):
    out = [] if out is None else out
    for k, v in ((schema or {}).get("properties") or {}).items():
        d = v.get("description", "")
        if v.get("type") == "object":
            _schema_hint(v, prefix + k + ".", out)
        elif v.get("type") == "array" and isinstance(v.get("items"), dict) and v["items"].get("type") == "object":
            out.append(f"- {prefix}{k}[] (list): {d}")
            _schema_hint(v["items"], prefix + k + "[].", out)
        else:
            out.append(f"- {prefix}{k}: {d}")
    return out

def _condense(context, question, schema=None):
    """Long transcripts: write notes per chunk (map), then answer from the notes (reduce)."""
    if len(context) <= MAX_DIRECT_CHARS:
        return context, (0, 0)
    key = hashlib.md5((context + question).encode()).hexdigest()
    if key in _notes_cache:
        return _notes_cache[key], (0, 0)
    chunks, cur = [], ""
    for line in context.splitlines(keepends=True):
        if len(cur) + len(line) > CHUNK_CHARS and cur:
            chunks.append(cur); cur = ""
        cur += line
    if cur:
        chunks.append(cur)
    hint = "\n".join(_schema_hint(schema))[:2500]
    hint_txt = f"\nThe final answer must fill these fields, so capture facts for them:\n{hint}\n" if hint else ""
    notes, tin, tout = [], 0, 0
    sysmsg = "You take careful notes on meeting transcripts."
    for i, ch in enumerate(chunks, 1):
        user = (f"Question to answer later: {question}\n{hint_txt}\nTranscript part {i} of {len(chunks)}:\n{ch}\n\n"
                "Write notes (max 450 words) relevant to the question: decisions, key facts and numbers, "
                "and action items (who - use speaker IDs, what, deadline). If nothing is relevant, write 'nothing relevant'.")
        text, u = _chat(sysmsg, user, 1100)
        notes.append(f"[Notes, part {i}/{len(chunks)}]\n{text.strip()}")
        tin += u[0]; tout += u[1]
    condensed = "\n\n".join(notes)
    _notes_cache[key] = condensed
    return condensed, (tin, tout)

def call_extract(context, question, schema, feedback=None):
    ctx, (ti, to) = _condense(context, question, schema)
    user = f"<context>\n{ctx}\n</context>\n\nQuestion: {question}"
    if feedback:
        user += f"\n\nYour previous answer had a problem: {feedback}\nFix it and answer again."
    pred, u = _chat_json(SYSTEM, user, schema, 6000)
    return pred, (u[0] + ti, u[1] + to)

def call_critic(context, question, prediction):
    ctx, (ti, to) = _condense(context, question)
    user = (f"<context>\n{ctx}\n</context>\n\nQuestion: {question}\n\nProposed answer (JSON):\n{prediction}\n\n"
            "Check every value in the answer against the context. List each value that is NOT supported by "
            "the context or contradicts it. If everything is supported, set grounded=true and issues=[].")
    v, u = _chat_json("You are a strict fact-checker. Return ONLY JSON.", user, CRITIC_SCHEMA, 1500)
    v = v if isinstance(v, dict) and "grounded" in v else {"grounded": True, "issues": []}
    return v, (u[0] + ti, u[1] + to)
