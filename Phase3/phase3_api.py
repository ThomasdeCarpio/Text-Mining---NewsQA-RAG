"""Local Gemini execution: standard-library HTTP, durable quota and result state.

Never import an SDK, retry implicitly, download a model, or persist credentials.
The shared SQLite ledger counts requests before sending, including interrupted
and failed requests. Limits are local ceilings, not a claim about account quota.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"


class RunPaused(RuntimeError):
    """Stop safely; completed results and consumed quota remain on disk."""


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


class QuotaLedger:
    def __init__(self, path, *, rpm=15, daily=500, reserve=25, clock=time.time, sleep=time.sleep):
        if type(rpm) is not int or not 1 <= rpm <= 15 or type(daily) is not int or not 1 <= daily <= 500:
            raise ValueError("Use positive local ceilings at or below 15 RPM / 500 per day")
        if type(reserve) is not int or not 0 <= reserve < daily:
            raise ValueError("Daily safety reserve must be between zero and daily limit minus one")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rpm, self.daily, self.reserve = rpm, daily, reserve
        self.clock, self.sleep = clock, sleep
        # Fail clearly if the host lacks timezone data; never silently reset in UTC.
        self.timezone = ZoneInfo("America/Los_Angeles")
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS requests (
                    id INTEGER PRIMARY KEY, started REAL NOT NULL, day TEXT NOT NULL,
                    run TEXT NOT NULL, case_id TEXT NOT NULL, policy TEXT NOT NULL,
                    outcome TEXT NOT NULL DEFAULT 'reserved', payload TEXT);
                CREATE INDEX IF NOT EXISTS request_case ON requests(run, case_id, policy);
                CREATE TABLE IF NOT EXISTS predictions (
                    run TEXT NOT NULL, case_id TEXT NOT NULL, policy TEXT NOT NULL,
                    payload TEXT NOT NULL, PRIMARY KEY(run, case_id, policy));
                CREATE TABLE IF NOT EXISTS settings (name TEXT PRIMARY KEY, value REAL NOT NULL);
                INSERT OR IGNORE INTO settings VALUES ('not_before', 0);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        finally:
            db.close()

    def day(self, timestamp):
        return datetime.fromtimestamp(timestamp, self.timezone).date().isoformat()

    def acquire(self, run, case_id, policy, *, max_attempts=3):
        while True:
            now = self.clock()
            with self.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                attempts = db.execute("SELECT COUNT(*) FROM requests WHERE run=? AND case_id=? AND policy=?", (run, case_id, policy)).fetchone()[0]
                if attempts >= max_attempts:
                    return None
                used = db.execute("SELECT COUNT(*) FROM requests WHERE day=?", (self.day(now),)).fetchone()[0]
                if used >= self.daily - self.reserve:
                    raise RunPaused("Local daily request budget exhausted; resume after midnight Pacific. Do not delete the quota ledger.")
                not_before = db.execute("SELECT value FROM settings WHERE name='not_before'").fetchone()[0]
                if not_before > now:
                    raise RunPaused("Provider cooldown is still active; resume later using the same quota ledger.")
                recent = [r[0] for r in db.execute("SELECT started FROM requests WHERE started>? ORDER BY started", (now - 60,))]
                last = db.execute("SELECT MAX(started) FROM requests").fetchone()[0]
                wait = max(0, (last or 0) + max(4.2, 60 / self.rpm + 0.2) - now)
                if len(recent) >= self.rpm:
                    wait = max(wait, recent[-self.rpm] + 60.2 - now)
                if wait <= 0:
                    cursor = db.execute("INSERT INTO requests(started,day,run,case_id,policy) VALUES (?,?,?,?,?)", (now, self.day(now), run, case_id, policy))
                    return cursor.lastrowid
            self.sleep(min(wait, 30))

    def finish(self, request_id, outcome, payload=None):
        with self.connect() as db:
            db.execute("UPDATE requests SET outcome=?,payload=? WHERE id=?", (outcome, json.dumps(payload) if payload is not None else None, request_id))

    def cooldown(self, seconds):
        with self.connect() as db:
            db.execute("UPDATE settings SET value=MAX(value,?) WHERE name='not_before'", (self.clock() + seconds,))

    def save_prediction(self, run, case_id, policy, prediction):
        with self.connect() as db:
            db.execute("INSERT INTO predictions VALUES (?,?,?,?)", (run, case_id, policy, json.dumps(prediction)))

    def prediction(self, run, case_id, policy):
        with self.connect() as db:
            row = db.execute("SELECT payload FROM predictions WHERE run=? AND case_id=? AND policy=?", (run, case_id, policy)).fetchone()
        return json.loads(row[0]) if row else None

    def predictions_for(self, run, policy):
        with self.connect() as db:
            rows = db.execute("SELECT case_id,payload FROM predictions WHERE run=? AND policy=?", (run, policy)).fetchall()
        return {case_id: json.loads(payload) for case_id, payload in rows}

    def completed_response(self, run, case_id, policy):
        # Recover terminal blocks too, including a crash before prediction export.
        with self.connect() as db:
            row = db.execute("SELECT payload FROM requests WHERE run=? AND case_id=? AND policy=? AND outcome IN ('success','provider_blocked') ORDER BY id DESC LIMIT 1", (run, case_id, policy)).fetchone()
        return json.loads(row[0]) if row else None

    def attempts(self, run, case_id, policy):
        with self.connect() as db:
            return db.execute("SELECT COUNT(*) FROM requests WHERE run=? AND case_id=? AND policy=?", (run, case_id, policy)).fetchone()[0]

    def usage(self, run):
        with self.connect() as db:
            records = db.execute("SELECT outcome,payload FROM requests WHERE run=?", (run,)).fetchall()
        result = {"http_attempts": len(records), "outcomes": {}, "reported_tokens_all_attempts": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}}
        for outcome, payload in records:
            result["outcomes"][outcome] = result["outcomes"].get(outcome, 0) + 1
            usage = (json.loads(payload).get("usage", {}) if payload else {})
            for key in result["reported_tokens_all_attempts"]:
                result["reported_tokens_all_attempts"][key] += usage.get(key, 0)
        result["notice"] = "Requests without a provider usage response may still consume quota/tokens. This ledger cannot see other applications in the same Google project."
        return result


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RunPaused("Unexpected API redirect; no credential forwarded")


def request_payload(settings, system_prompt, user_prompt):
    return {"model": settings["model"], "temperature": settings["temperature"],
            "max_tokens": settings["max_tokens"], "reasoning_effort": settings["reasoning_effort"],
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]}


def gemini_http(payload, api_key, *, timeout=60):
    if not api_key or not api_key.strip():
        raise RunPaused("Set GEMINI_API_KEY privately in your environment")
    request = urllib.request.Request(ENDPOINT, data=json.dumps(payload).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key.strip()})
    # urllib performs no automatic application-level retry. Reject redirects.
    with urllib.request.build_opener(NoRedirect()).open(request, timeout=timeout) as response:
        return json.load(response)


def generate_one(ledger, run, case_id, policy, payload, parse, api_key, *, transport=gemini_http, max_attempts=3):
    recovered = ledger.completed_response(run, case_id, policy)
    if recovered:
        return recovered
    while True:
        request_id = ledger.acquire(run, case_id, policy, max_attempts=max_attempts)
        if request_id is None:
            return {"status": "exhausted", "answerability": None, "answer": None, "citations": [], "error": "attempt_limit_reached"}
        started = ledger.clock()
        usage = {}
        raw_text = None
        finish_reason = None
        try:
            response = transport(payload, api_key)
            raw_usage = response.get("usage") or {}
            for dest, src in (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens"), ("total_tokens", "total_tokens")):
                value = raw_usage.get(src)
                if type(value) is int and value >= 0:
                    usage[dest] = value
            choice = response["choices"][0]
            # Gemini can return a content_filter finish reason without a message.
            # Record the reason before reading content; a provider block is not
            # an abstention, nor a transient format error worth repeating.
            finish_reason = choice.get("finish_reason")
            if isinstance(finish_reason, str) and finish_reason.partition(":")[0].strip() == "content_filter":
                result = {"status": "provider_blocked", "answerability": None, "answer": None, "citations": [],
                          "error": "provider_content_filter", "finish_reason": finish_reason, "usage": usage,
                          "generation_ms": round((ledger.clock() - started) * 1000, 1)}
                ledger.finish(request_id, "provider_blocked", result)
                return result
            text = choice["message"]["content"]
            raw_text = text if isinstance(text, str) else None
            if finish_reason != "stop" or not isinstance(text, str) or not text.strip():
                raise ValueError("Incomplete/empty generation")
            parsed = parse(text)
            result = {**parsed, "status": "success", "usage": usage, "generation_ms": round((ledger.clock() - started) * 1000, 1)}
            ledger.finish(request_id, "success", result)
            return result
        except urllib.error.HTTPError as exc:
            ledger.finish(request_id, f"http_{exc.code}")
            if exc.code == 429:
                try:
                    delay = float(exc.headers.get("Retry-After", "60"))
                except (TypeError, ValueError):
                    delay = 60
                ledger.cooldown(max(60, delay) if math.isfinite(delay) else 60)
                raise RunPaused("Gemini returned 429; paused with quota and results saved. Check project RPM/TPM/daily capacity before resuming.") from None
            if exc.code not in (408, 500, 502, 503, 504):
                raise RunPaused(f"Gemini HTTP {exc.code}; verify credentials/model/access. Response details are not logged.") from None
            ledger.sleep(min(5 * ledger.attempts(run, case_id, policy), 30))
        except (ValueError, KeyError, IndexError, TypeError):
            ledger.finish(request_id, "invalid_response", {"usage": usage, "raw_text": raw_text, "finish_reason": finish_reason})
        except (urllib.error.URLError, TimeoutError, OSError):
            ledger.finish(request_id, "transport_error")
            ledger.sleep(5)
        except BaseException:
            ledger.finish(request_id, "interrupted_or_unexpected")
            raise


def export_predictions(ledger, run, cases, policy, path):
    by_id = ledger.predictions_for(run, policy)
    if set(by_id) - {case["case_id"] for case in cases}:
        raise ValueError("Unexpected case IDs in the durable result ledger")
    records = [by_id[case["case_id"]] for case in cases if case["case_id"] in by_id]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return records
