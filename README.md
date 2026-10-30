# decision-model-evals

Ranks [OpenRouter](https://openrouter.ai) decision models, and text models for
comparison, by log score on 100 resolved binary questions from
[Metaculus](https://www.metaculus.com).

For a decision model, each question title is sent to OpenRouter's Decisions API
(`/api/alpha/decisions`) as a `noul`, the API's yes/no question type:

```json
{
  "model": "typesafe/jev-1.14",
  "state": {"text": "Trump reelected by November 10th 2020"},
  "questions": {
    "resolved_yes": {
      "type": "noul",
      "instructions": "The Metaculus forecasting question in `text` has already resolved. Did it resolve YES?"
    }
  }
}
```

The state key is `text` because ScaleDown only accepts a state object with `text`,
`document` and `document_mime_type` keys. The model returns the probability that
the question resolved YES. Models are scored
by the mean natural log of the probability they gave to the actual outcome.

### Text models

A text model gets the same question through chat completions
(`/api/v1/chat/completions`), and has to answer with the probability in JSON
matching a schema:

```json
{
  "model": "mistralai/mistral-nemo",
  "messages": [{"role": "user", "content": "The Metaculus forecasting question below has already resolved. What is the probability that it resolved YES?\n\nTrump reelected by November 10th 2020"}],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "forecast",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {"probability": {"type": "number", "description": "Probability that the question resolved YES, from 0 to 1"}},
        "required": ["probability"],
        "additionalProperties": false
      }
    }
  },
  "provider": {"require_parameters": true}
}
```

Text models answer without reasoning first. A model that can reason also gets
`"reasoning": {"enabled": false}`. That isn't sent to models that can't reason,
because then no provider would accept the request. A model whose reasoning can't be
turned off gets the lowest effort it supports instead, e.g.
`"reasoning": {"effort": "minimal"}` for GPT-5 Nano. Models with no effort setting
are rejected before the run starts. Neither setting always stops the reasoning. Even
with reasoning off, some models still report a few dozen reasoning tokens; Mercury 2.5
does, for example. At its lowest effort (`low`), GLM Flash can still use hundreds.

`require_parameters` makes OpenRouter route only to providers that support every
parameter in the request, so the schema and the reasoning setting can't be silently
ignored. Some providers enforce
the schema exactly. Others only use it as a strong hint, so a response without a valid
probability between 0 and 1 counts as malformed and is retried.

A model that doesn't support structured outputs gets no `response_format`. Instead,
the schema is added to the end of the prompt:

```
Answer with only a JSON object matching this JSON schema:
{"type": "object", "properties": {"probability": {"type": "number", "description": "Probability that the question resolved YES, from 0 to 1"}}, "required": ["probability"], "additionalProperties": false}
```

The reply has to be that JSON object and nothing else. Anything else counts as
malformed and is retried, in the same way.

This is a different measurement from a decision model's: the text model writes a
probability out as a number. The leaderboard marks each model's `api` as `decisions`
or `chat` so the two kinds aren't mistaken for each other.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```sh
cp .env.example .env   # then put your OpenRouter API key in .env
```

uv installs the dependencies (`httpx`, `python-dotenv`, `tqdm`) on the first run.

## Usage

```sh
uv run evaluate.py MODEL [MODEL ...]     # evaluate models, skipping ones already done
uv run evaluate.py -f models.txt         # evaluate every model listed in a file
uv run evaluate.py MODEL --force         # discard a model's results and re-evaluate it
uv run evaluate.py                       # print the leaderboard
```

Model ids are OpenRouter model ids, e.g. `typesafe/jev-1.14`. Before it starts,
the script looks each one up in OpenRouter's model list. Decision models are asked
through the Decisions API, and text models through chat completions. A text model
must be able to turn reasoning off or choose an effort. OpenRouter filters the model list by your account's
privacy and provider settings, so a model those settings block is rejected even if it
is on the public list. In a models file, put one id per line; `#` starts a comment.

`-j N` sets the number of parallel requests per model (default 8).

A progress bar per model, showing failures and running cost, goes to stderr. The
leaderboard is printed after every run.

### Inspecting raw requests and responses

```sh
uv run samples.py MODEL                  # 3 random questions
uv run samples.py MODEL -n 10            # 10 random questions
uv run samples.py MODEL --ids 1100 9939  # specific questions, by Metaculus id
```

This sends the same requests as `evaluate.py`, once each and with no retries. It
writes each question with its exact request body and the raw response (status,
time taken and body) to `samples/MODEL.json`, replacing any earlier samples for
that model. Errors are saved as they came back, which makes this handy for
debugging a model whose questions keep failing.

### Cost against score

```sh
uv run frontier.py           # write frontier.html
uv run frontier.py --open    # write it and open it in a browser
```

This writes `frontier.html`, a standalone page that plots each model's log score
against the average cost per question that OpenRouter charged (`usage.cost` in its
results). Models on the Pareto frontier are highlighted and joined by a step line.
A model is on the frontier when no other model is at least as cheap and at least
as good. Hover over a dot, or tab to it, for details. "Show table" lists every
model. Free (`:free`) models take the cheap end of the frontier, so a checkbox
leaves them out and recomputes it.

## Scoring

```
rank  log score  missing  api        model
        -0.6931        0             (always 50%)
```

- **log score** is the mean of ln(p) over all 100 questions, where p is the
  probability given to the outcome that happened. Higher is better; 0 is perfect.
- Probabilities are clipped to [0.1%, 99.9%], the range Metaculus accepts, so a
  single confident miss can't send a score to −∞.
- **missing** counts questions where every request failed. These are scored as 50%.
- **api** is `decisions` for a decision model and `chat` for a text model (see
  [Text models](#text-models)).
- The `(always 50%)` row is a baseline. 64 of the 100 questions resolved NO, so a
  model that always answers 36% YES would score about −0.653, beating that baseline
  without knowing anything about the individual questions.

The model only sees the question title. Many of the questions resolved before
current models' training cutoffs, so the score measures what a model knows about
past events as well as how well it forecasts.

## Files

- `evaluate.py`: the whole evaluation: querying, retries, result storage and the
  leaderboard.
- `samples.py`: saves raw requests and responses (see above). The results go in
  `samples/`.
- `frontier.py` and `frontier_template.html`: the cost-against-score chart (see
  above). `frontier.py` fills the template with data and writes `frontier.html`.
- `questions.json`: the dataset. These are the top 100 resolved binary Metaculus
  questions by number of forecasters, retrieved 2026-10-27. Each has an `id`,
  `title`, `resolution` (`yes`/`no`), forecaster count and `url`. Questions
  resolved Annulled or Ambiguous were skipped and are listed under `excluded`.
- `results/`: created on the first run. It holds one `.jsonl` file per model, with
  `/` in the model id replaced by `__`. Each line is one question's result: the
  probability of YES, the API it was asked through (`decisions` or `chat`; results
  saved before text models were supported have no `api` field and are all
  `decisions`), the model and provider that served it, token usage and cost, and a
  timestamp. Lines are written as answers arrive, so an interrupted run picks up
  where it left off. A rerun only retries questions that are missing or failed.

## Errors

Network errors, timeouts, rate limits (429), 5xx responses and malformed responses
get up to 4 attempts in total, with exponential backoff between them. Other HTTP
errors are recorded as failures without retrying. A 401 (bad key) or 402 (out of credits) stops the run
immediately. Any results saved so far are kept.
