"""Plot log score, Brier score or accuracy against OpenRouter cost, with the Pareto
frontier highlighted.

    uv run frontier.py           write frontier.html
    uv run frontier.py --open    write it and open it in a browser

Cost is the mean of what OpenRouter charged per answered question (usage.cost in
results/). A model is on the frontier if no other model is at least as cheap and at
least as good. The solid frontier is over paid models only; a dotted line shows the
frontier with free models counted too. The page has a checkbox to leave free models out,
one to show only models with open weights (from licences.json), one per model creator to
show only some creators' models, and a setting to switch between log score, Brier score
and accuracy.
"""

import argparse
import json
import math
import statistics
import sys
import webbrowser

from evaluate import ROOT, accuracy, all_results, brier_score, load_questions, score

TEMPLATE = ROOT / "frontier_template.html"
OUTPUT = ROOT / "frontier.html"
LICENCES = ROOT / "licences.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--open", action="store_true", help="open the page in a browser")
    args = parser.parse_args()

    questions = load_questions()
    licences = json.loads(LICENCES.read_text(encoding="utf-8"))["models"]
    models = []
    for model, results in all_results().items():
        costs = [record["usage"]["cost"] for record in results.values()
                 if record.get("probability") is not None
                 and (record.get("usage") or {}).get("cost") is not None]
        if not costs:
            print(f"{model}: no cost recorded, leaving it out", file=sys.stderr)
            continue
        if model not in licences:
            print(f"{model}: not in {LICENCES.name}, counting its weights as unknown",
                  file=sys.stderr)
        licence = licences.get(model, {})
        mean, missing = score(results, questions)
        models.append({"model": model, "log": mean, "brier": brier_score(results, questions),
                       "accuracy": accuracy(results, questions), "missing": missing,
                       "cost": statistics.fmean(costs),
                       "weights": licence.get("weights", "unknown"),
                       "licence": licence.get("licence")})

    # Keyed like the models' scores: what always answering 50% gets.
    data = {"questions": len(questions),
            "baselines": {"log": math.log(0.5), "brier": 0.25, "accuracy": 0.5},
            "models": models}
    # Escape "<" so no model id can close the <script> element the data sits in.
    blob = json.dumps(data).replace("<", "\\u003c")
    OUTPUT.write_text(TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", blob),
                      encoding="utf-8")
    print(OUTPUT)
    if args.open:
        webbrowser.open(OUTPUT.as_uri())


if __name__ == "__main__":
    main()
