"""Rank OpenRouter models by log score on 100 resolved Metaculus questions.

Each question is posed to decision models through OpenRouter's Decisions API as a
Noul (yes/no) question, and the model returns the probability that it resolved YES.
Text models are asked through chat completions instead, and must state the
probability in JSON matching a schema, with reasoning off or, for models that can't
turn it off, at their lowest effort. The schema is enforced through structured outputs
where the model supports them, and otherwise given in the prompt.

    uv run evaluate.py MODEL [MODEL ...]     evaluate models, skipping ones already done
    uv run evaluate.py -f models.txt         evaluate every model listed in a file
    uv run evaluate.py MODEL --force         discard a model's results and re-evaluate it
    uv run evaluate.py                       print the leaderboard

Results are saved per model in results/, one JSON line per question, so an
interrupted run picks up where it left off.
"""

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import httpx
from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
QUESTIONS_FILE = ROOT / "questions.json"
RESULTS_DIR = ROOT / "results"
OPENROUTER_URL = "https://openrouter.ai/api"
DECISIONS_PATH = "/alpha/decisions"
CHAT_PATH = "/v1/chat/completions"

# Metaculus only accepts binary forecasts between 0.1% and 99.9%; clip to the same
# range so one overconfident miss (decision models do return 0s) can't send a
# model's score to -infinity.
P_MIN, P_MAX = 0.001, 0.999
MAX_ATTEMPTS = 4
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504, 524, 529}
FATAL_STATUS = {401, 402}  # bad key / out of credits: no point continuing

# The question title goes in state as {"text": ...}: ScaleDown only accepts a state
# object with keys text, document and document_mime_type.
QUESTION = {
    "type": "noul",
    "instructions": "The Metaculus forecasting question in `text` has already resolved. "
                    "Did it resolve YES?",
}

# Text models get the question as a prompt and must answer in this schema, without
# reasoning first, or with as little as the model allows. require_parameters stops
# OpenRouter routing to a provider that would ignore the schema or the reasoning
# setting. Models without structured outputs are told the schema in the prompt instead.
CHAT_PROMPT = ("The Metaculus forecasting question below has already resolved. "
               "What is the probability that it resolved YES?\n\n{title}")
FORECAST_SCHEMA = {
    "name": "forecast",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"probability": {
            "type": "number",
            "description": "Probability that the question resolved YES, from 0 to 1"}},
        "required": ["probability"],
        "additionalProperties": False,
    },
}
SCHEMA_PROMPT = ("\n\nAnswer with only a JSON object matching this JSON schema:\n"
                 + json.dumps(FORECAST_SCHEMA["schema"]))
# OpenRouter's reasoning efforts, lowest first.
EFFORTS = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]


class FatalError(Exception):
    pass


class Route(NamedTuple):
    """How a model is asked: "decisions" or "chat". For a chat model that can reason,
    the reasoning parameter that turns it off or down, and whether the model supports
    structured outputs; if it doesn't, the schema goes in the prompt."""
    api: str
    reasoning: dict | None = None
    structured: bool = True


def make_client() -> httpx.Client:
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY is not set; put it in .env (see .env.example)")
    return httpx.Client(base_url=OPENROUTER_URL, timeout=120,
                        headers={"Authorization": f"Bearer {api_key}"})


def model_routes(client: httpx.Client, models: list[str]) -> dict[str, Route]:
    """Map each model id to how it's asked: through the Decisions API for decision
    models, through chat completions for text models that can turn reasoning off or
    choose an effort. Exits on any other id."""
    def listed(**params) -> dict[str, dict]:
        resp = client.get("/v1/models", params=params)
        return {m["id"]: m for m in resp.raise_for_status().json()["data"]}

    decision_models, text_models = listed(output_modalities="decisions"), listed()
    routes, problems = {}, []
    for model in models:
        if model in decision_models:
            routes[model] = Route("decisions")
            continue
        info = text_models.get(model)
        # Models that can't reason have no "reasoning" field.
        reasoning = (info or {}).get("reasoning")
        structured = "structured_outputs" in (info or {}).get("supported_parameters", [])
        if info is None:
            problems.append(f"{model}: not a decision or text model this OpenRouter key "
                            "can use (the list is filtered by the account's settings)")
        elif reasoning is None:
            routes[model] = Route("chat", None, structured)
        elif not reasoning.get("mandatory"):
            routes[model] = Route("chat", {"enabled": False}, structured)
        else:
            # Reasoning can't be turned off, so ask for the lowest effort it has.
            efforts = [e for e in EFFORTS if e in reasoning.get("supported_efforts", [])]
            if efforts:
                routes[model] = Route("chat", {"effort": efforts[0]}, structured)
            else:
                problems.append(f"{model}: text model that can't turn reasoning off "
                                "or choose an effort")
    if problems:
        sys.exit("\n".join(problems))
    return routes


def request(route: Route, model: str, question: dict) -> tuple[str, dict]:
    """The API path and request body that ask a model one question."""
    if route.api == "chat":
        body = {"model": model,
                "messages": [{"role": "user",
                              "content": CHAT_PROMPT.format(title=question["title"])}],
                "provider": {"require_parameters": True}}
        if route.structured:
            body["response_format"] = {"type": "json_schema", "json_schema": FORECAST_SCHEMA}
        else:
            body["messages"][0]["content"] += SCHEMA_PROMPT
        # Only for models that can reason: with require_parameters, sending it to one
        # that can't leaves no provider to route to.
        if route.reasoning:
            body["reasoning"] = route.reasoning
        return CHAT_PATH, body
    return DECISIONS_PATH, {"model": model, "state": {"text": question["title"]},
                            "questions": {"resolved_yes": QUESTION}}


def probability(api: str, data: dict) -> float | None:
    """The probability of YES in a response body, or None if there isn't a valid one."""
    try:
        if api == "chat":
            p = json.loads(data["choices"][0]["message"]["content"])["probability"]
        else:
            p = data["answers"]["resolved_yes"]["noul"]
        p = float(p)
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return p if 0 <= p <= 1 else None


def forecast(client: httpx.Client, model: str, route: Route, question: dict) -> dict:
    path, payload = request(route, model, question)
    error = None
    for attempt in range(MAX_ATTEMPTS):
        if attempt:
            time.sleep(2**attempt)
        try:
            resp = client.post(path, json=payload)
        except httpx.HTTPError as e:
            error = f"{type(e).__name__}: {e}"
            continue
        if resp.status_code in FATAL_STATUS:
            raise FatalError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        if resp.status_code != 200:
            error = f"HTTP {resp.status_code}: {resp.text[:300]}"
            if resp.status_code in RETRYABLE_STATUS:
                continue
            break
        data = resp.json()
        p = probability(route.api, data)
        if p is None:
            error = f"unexpected response: {resp.text[:300]}"
            continue
        return {"question_id": question["id"], "probability": p, "api": route.api,
                "served_by": data.get("model"), "provider": data.get("provider"),
                "usage": data.get("usage")}
    return {"question_id": question["id"], "probability": None, "api": route.api,
            "error": error}


def results_path(model: str) -> Path:
    return RESULTS_DIR / (model.replace("/", "__") + ".jsonl")


def load_results(path: Path) -> dict[int, dict]:
    """Map question id -> latest record for that question."""
    results = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                results[record["question_id"]] = record
    return results


def evaluate_model(client: httpx.Client, model: str, route: Route, questions: list[dict],
                   concurrency: int, force: bool) -> None:
    path = results_path(model)
    if force:
        path.unlink(missing_ok=True)
    done = load_results(path)
    todo = [q for q in questions if done.get(q["id"], {}).get("probability") is None]
    if not todo:
        print(f"{model}: already evaluated, skipping (use --force to redo)", file=sys.stderr)
        return

    RESULTS_DIR.mkdir(exist_ok=True)
    failed, cost = 0, 0.0
    with (ThreadPoolExecutor(concurrency) as pool, path.open("a", encoding="utf-8") as f,
          tqdm(total=len(todo), desc=model, unit="q") as bar):
        futures = [pool.submit(forecast, client, model, route, q) for q in todo]
        try:
            for future in as_completed(futures):
                record = future.result()
                record["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                if record["probability"] is None:
                    failed += 1
                    tqdm.write(f"  question {record['question_id']} failed: {record['error']}",
                               file=sys.stderr)
                cost += (record.get("usage") or {}).get("cost") or 0
                bar.set_postfix(failed=failed, cost=f"${cost:.3f}")
                bar.update()
        except FatalError:
            for future in futures:
                future.cancel()
            raise


def load_questions() -> list[dict]:
    return json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))["questions"]


def all_results() -> dict[str, dict[int, dict]]:
    """Map model id -> its results, for every model with results saved."""
    models = {}
    for path in sorted(RESULTS_DIR.glob("*.jsonl")):
        results = load_results(path)
        if results:
            models[path.stem.replace("__", "/")] = results
    return models


def results_api(results: dict[int, dict]) -> str:
    """The API a model's results came from. Results saved before text models were
    supported have no "api" field, and are all from the Decisions API."""
    return next(iter(results.values())).get("api", "decisions")


def score(results: dict[int, dict], questions: list[dict]) -> tuple[float, int]:
    """Mean log score, and the number of unanswered questions (scored as 50%)."""
    total, missing = 0.0, 0
    for q in questions:
        p = results.get(q["id"], {}).get("probability")
        if p is None:
            missing += 1
            p = 0.5
        p = min(max(p, P_MIN), P_MAX)
        total += math.log(p if q["resolution"] == "yes" else 1 - p)
    return total / len(questions), missing


def leaderboard(questions: list[dict]) -> None:
    rows = []
    for model, results in all_results().items():
        mean, missing = score(results, questions)
        rows.append((mean, model, missing, results_api(results)))
    baseline = "(always 50%)"
    rows.append((math.log(0.5), baseline, 0, ""))
    rows.sort(reverse=True)

    print(f"\nMean log score over {len(questions)} questions "
          "(ln of probability given to the actual outcome; higher is better, 0 is perfect).")
    print("Questions with no answer (request kept failing) are scored as 50%.")
    print("api: decisions = a decision model's answer from the Decisions API; "
          "chat = a probability a text model stated in JSON.\n")
    print(f"{'rank':>4}  {'log score':>9}  {'missing':>7}  {'api':<9}  model")
    rank = 0
    for mean, model, missing, api in rows:
        label = ""
        if model != baseline:
            rank += 1
            label = rank
        print(f"{label:>4}  {mean:9.4f}  {missing:7}  {api:<9}  {model}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*",
                        help="OpenRouter model ids: decision models, e.g. typesafe/jev-1.14, "
                             "or text models that can turn reasoning off or choose an "
                             "effort")
    parser.add_argument("-f", "--models-file", type=Path,
                        help="file with one model id per line (# starts a comment)")
    parser.add_argument("--force", action="store_true",
                        help="discard existing results for the given models and re-evaluate")
    parser.add_argument("-j", "--concurrency", type=int, default=8,
                        help="parallel requests per model (default 8)")
    args = parser.parse_args()

    models = list(args.models)
    if args.models_file:
        for line in args.models_file.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                models.append(line)
    models = list(dict.fromkeys(models))
    questions = load_questions()

    if models:
        client = make_client()
        routes = model_routes(client, models)
        for model in models:
            try:
                evaluate_model(client, model, routes[model], questions, args.concurrency,
                               args.force)
            except FatalError as e:
                sys.exit(f"{model}: {e}")

    leaderboard(questions)


if __name__ == "__main__":
    main()
