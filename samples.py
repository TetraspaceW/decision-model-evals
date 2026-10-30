"""Save raw API requests and responses for a model, for inspection.

    uv run samples.py MODEL                  3 random questions
    uv run samples.py MODEL -n 10            10 random questions
    uv run samples.py MODEL --ids 1100 9939  specific questions, by Metaculus id

Each request is the same one evaluate.py sends, but is sent once with no retries,
so errors are kept as they came back. Samples are written to samples/MODEL.json,
replacing any earlier samples for that model.
"""

import argparse
import json
import random
import sys

import httpx
from tqdm import tqdm

from evaluate import OPENROUTER_URL, ROOT, load_questions, make_client, model_routes, request

SAMPLES_DIR = ROOT / "samples"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="OpenRouter model id, e.g. typesafe/jev-1.14")
    parser.add_argument("-n", type=int, default=3,
                        help="number of random questions to send (default 3)")
    parser.add_argument("--ids", type=int, nargs="+",
                        help="Metaculus ids of the questions to send, instead of random ones")
    args = parser.parse_args()

    questions = load_questions()
    if args.ids:
        by_id = {q["id"]: q for q in questions}
        unknown = [str(i) for i in args.ids if i not in by_id]
        if unknown:
            sys.exit(f"Not in questions.json: {', '.join(unknown)}")
        chosen = [by_id[i] for i in args.ids]
    else:
        chosen = random.sample(questions, min(args.n, len(questions)))

    client = make_client()
    route = model_routes(client, [args.model])[args.model]
    samples = []
    for question in tqdm(chosen, desc=args.model, unit="q"):
        path, body = request(route, args.model, question)
        sample = {"question": question,
                  "request": {"url": OPENROUTER_URL + path, "body": body}}
        try:
            resp = client.post(path, json=body)
        except httpx.HTTPError as e:
            sample["error"] = f"{type(e).__name__}: {e}"
            tqdm.write(f"  question {question['id']}: {sample['error']}", file=sys.stderr)
        else:
            try:
                body = resp.json()
            except ValueError:
                body = resp.text
            sample["response"] = {"status": resp.status_code,
                                  "elapsed_s": round(resp.elapsed.total_seconds(), 2),
                                  "body": body}
            if resp.status_code != 200:
                tqdm.write(f"  question {question['id']}: HTTP {resp.status_code}",
                           file=sys.stderr)
        samples.append(sample)

    SAMPLES_DIR.mkdir(exist_ok=True)
    path = SAMPLES_DIR / (args.model.replace("/", "__") + ".json")
    path.write_text(json.dumps(samples, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
