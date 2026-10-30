"""Plot log score against OpenRouter cost, with the Pareto frontier highlighted.

    uv run frontier.py           write frontier.html
    uv run frontier.py --open    write it and open it in a browser

Cost is the mean of what OpenRouter charged per answered question (usage.cost in
results/). A model is on the frontier if no other model is at least as cheap and at
least as good. The solid frontier is over paid models only; a dotted line shows the
frontier with free models counted too. The page has a checkbox to leave free models out.
"""

import argparse
import json
import math
import statistics
import sys
import webbrowser

from evaluate import ROOT, all_results, load_questions, score

TEMPLATE = ROOT / "frontier_template.html"
OUTPUT = ROOT / "frontier.html"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--open", action="store_true", help="open the page in a browser")
    args = parser.parse_args()

    questions = load_questions()
    models = []
    for model, results in all_results().items():
        costs = [record["usage"]["cost"] for record in results.values()
                 if record.get("probability") is not None
                 and (record.get("usage") or {}).get("cost") is not None]
        if not costs:
            print(f"{model}: no cost recorded, leaving it out", file=sys.stderr)
            continue
        mean, missing = score(results, questions)
        models.append({"model": model, "score": mean, "missing": missing,
                       "cost": statistics.fmean(costs)})

    data = {"questions": len(questions), "baseline": math.log(0.5), "models": models}
    # Escape "<" so no model id can close the <script> element the data sits in.
    blob = json.dumps(data).replace("<", "\\u003c")
    OUTPUT.write_text(TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", blob),
                      encoding="utf-8")
    print(OUTPUT)
    if args.open:
        webbrowser.open(OUTPUT.as_uri())


if __name__ == "__main__":
    main()
