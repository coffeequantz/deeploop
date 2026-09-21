# deeploop

**A budget-first autonomous agent harness that loops cheap models until verifiable success criteria actually pass.**

DeepSeek and other low-cost models make long agent loops economically viable — but cheap loops fail in expensive ways: they spin forever, declare victory without evidence, or quietly burn budget. deeploop is built around three rules:

1. **The harness owns the loop, never the model.** A separate verifier decides when the mission is done. The executor's "I'm finished!" is not evidence.
2. **Budgets and permissions are enforced in code, not in prompts.** Spend, iterations, wall clock, commands, paths and network access are checked at the execution layer. Prompt-level limits are suggestions; these are hard.
3. **Every iteration is checkpointed, logged, and resumable.** Per-iteration git commits, an append-only JSONL ledger, and a state snapshot mean a crash or Ctrl-C resumes instead of restarting.

```
task contract (YAML)
      │
      ▼
┌─────────────┐   plan    ┌──────────────┐   tools   ┌─────────────┐
│  controller │ ────────► │   executor   │ ────────► │ shell/files │
│  (budget,   │           │ (cheap model)│           │   /git      │
│ permissions,│ ◄──────── └──────────────┘           └─────────────┘
│  stuck det.)│   observe
│             │   verify   ┌──────────────────────────────────────┐
│             │ ────────►  │ ladder: command checks → LLM judge   │
│             │ ◄────────  │ (independent, evidence-only)         │
└─────────────┘            └──────────────────────────────────────┘
      │
      ▼
 ledger.jsonl · state.json · git checkpoints · TUI / headless
```

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/coffeequantz/deeploop/main/install.sh | sh
```

The installer is a short, readable shell script: it prefers `uv`, then `pipx`, and otherwise creates a private venv under `~/.local/share/deeploop` with a symlink in `~/.local/bin`. Then:

```bash
export DEEPSEEK_API_KEY=sk-...
deeploop --version
```

Other ways in:

```bash
uv tool install git+https://github.com/coffeequantz/deeploop    # or: pipx install git+...
pip install "deeploop-cli @ git+https://github.com/coffeequantz/deeploop"
```

Once a release is on PyPI: `uv tool install deeploop-cli`, `pipx install deeploop-cli`, or `pip install deeploop-cli`. The distribution is named `deeploop-cli` because `deeploop` was already taken on PyPI by an unrelated project; the command is still `deeploop`.

From a checkout, for development:

```bash
git clone https://github.com/coffeequantz/deeploop && cd deeploop
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
./install.sh --local          # same thing without activating a venv
```

Python 3.9+ is supported. The only runtime dependencies are `textual`, `pydantic`, `PyYAML` and `httpx`.

Maintainers: publishing runs through `.github/workflows/publish.yml` on GitHub release using PyPI trusted publishing — configure `deeploop-cli` as the project name and the `pypi` environment in the repo settings, then publish a release.

## Quickstart

```bash
deeploop brief brief/        # derive a mission contract from a brief folder, review it
deeploop run brief/          # run the loop until the verifier says done
deeploop report brief/       # status, criteria, spend per iteration
```

That opens the TUI: mission log in the center, enforced budget/criteria/permissions in the sidebar, one input line for human interjection or escalation answers.

## Start from a brief folder

Write the goal the way you'd explain it to a person, drop in whatever context matters, and point the harness at the folder:

```
brief/
  BRIEF.md            # narrative goal, constraints, non-goals
  context/*.md        # reference docs the agent may consult
  assets/*.png        # images: references to interpret, or files to use
```

```bash
deeploop brief brief/          # derive mission.yaml from the folder, review it, write it
deeploop run brief/            # same, then run the mission
```

`deeploop run <folder>` accepts a folder directly: if it contains a `mission.yaml` it runs it, and if it only has a `BRIEF.md` it derives the contract first. Derivation is interactive — you see the proposed goal, criteria, workspace and budget and can:

- **accept** it (writes `mission.yaml`, then the loop starts),
- **revise** it by typing a note ("split the tests criterion in two", "budget is too small"), which regenerates the proposal,
- **edit** the YAML directly in `$EDITOR` (console mode), or
- cancel, spending only the proposal call.

The proposal prompt insists on machine-checkable criteria and clamps budget/permissions to conservative values; you approve the result before anything runs. `--yes` accepts without review (useful in CI), and `deeploop brief --yes` only writes the contract.

Inside a mission, the brief is context, not command:

- `BRIEF.md` and the file manifest (paths, sizes, first lines) are injected into the planner and actor prompts. Context documents are listed, not stuffed — the actor opens them with `read_file` when relevant, which keeps per-iteration cost flat.
- The brief folder is added to `read_paths` automatically and never to `write_paths`, so a brief outside the workspace stays read-only. If your brief *is* your project folder, set `workspace: "."` and the agent works in place.
- Brief file hashes are recorded in the ledger at mission start, so you can prove what the agent saw; if the brief changes between runs, the harness flags it.
- Images: DeepSeek's chat models can't see them, so point `model.vision` at a vision-capable model (any OpenAI-compatible one, e.g. via OpenRouter) and each image is described once, cached by file hash in `.deeploop/artifacts/image_descriptions.json`, and folded into the text manifest. Without a vision model you get an explicit warning that the images were not included.

## Writing a mission

`deeploop init` writes a starter contract. The full reference:

```yaml
goal: "Make the test suite pass without weakening the tests."

success_criteria:               # at least one; prefer machine-checkable
  - id: tests-pass
    description: "pytest exits 0"
    check: "pytest -q"          # kind defaults to command
  - id: tests-not-weakened
    description: "Existing tests were not deleted or loosened."
    kind: judge                 # prose criterion: judged from raw evidence

budget:                         # hard limits, checked outside the model
  max_usd: 2.00
  max_iterations: 30
  max_wall_clock_minutes: 30

permissions:                    # enforced at the tool execution layer
  workspace: "."
  read_paths: ["."]
  write_paths: ["."]
  network: false                # best-effort proxy block; use a container for hard isolation
  allow_all_commands: false
  allow_commands: ["pytest", "python3", "git", "ls", "cat", "rg", "sed", "make"]
  deny_commands: ["sudo", "git push", "rm -rf /", "curl", "wget"]
  require_approval: []          # tools that always need a human answer

limits:
  max_tool_calls_per_iteration: 8
  command_timeout_seconds: 300
  max_output_chars: 6000
  stuck_after: 3                # consecutive no-change/no-progress iterations
  max_replans: 2

escalation:
  on_budget_exhausted: halt     # halt | ask | degrade
  on_stuck: ask                 # halt | ask | replan
  on_permission_denied: deny    # halt | ask | deny

provider:
  name: deepseek                # deepseek | openrouter | ollama | mock
  api_key_env: DEEPSEEK_API_KEY
  pricing: {}                   # optional per-model USD/1M token overrides

model:
  planner: deepseek-reasoner    # no tools required
  actor: deepseek-chat          # must support function calling
  critic: deepseek-chat
  judge: deepseek-reasoner
  vision: qwen/qwen-2.5-vl-72b-instruct  # optional: describe brief images once
  fallback_actor: deepseek-chat # used when on_budget_exhausted=degrade

checkpoint:
  enabled: true
  branch: deeploop/mission      # work on a branch so your main stays clean

brief:                          # optional context folder (see the brief section above)
  dir: "brief"
  describe_images: true
  max_context_chars: 6000
```

Validate before spending anything:

```bash
deeploop validate mission.yaml   # prints the contract and design warnings
```

Warnings are worth reading: prose-only criteria, `allow_all_commands`, network access, disabled checkpointing, and unusually large budgets all get flagged.

## CLI

| command | what it does |
| --- | --- |
| `deeploop init [dir]` | write a starter `mission.yaml` |
| `deeploop brief [dir]` | derive a contract from a brief folder, review it, write `mission.yaml` |
| `deeploop validate <contract>` | parse and sanity-check a contract |
| `deeploop run <contract-or-folder>` | run a mission (TUI when interactive, `--headless` otherwise) |
| `deeploop resume <contract>` | continue from the ledger + state snapshot |
| `deeploop report [dir]` | summarize status, criteria, spend by iteration |
| `deeploop models` | pricing table used for budget accounting |

Exit codes: `0` done, `2` halted/budget-exhausted, `1` error — usable directly in CI.

Useful flags: `--provider mock|deepseek|openrouter|ollama`, `--headless`, `--verbose`, `--yes`.

## How completion is decided

The verification ladder runs cheapest-first, and the actor is never the judge:

1. **Deterministic checks** (`kind: command`) run the real command in the workspace. Exit code decides. No LLM involved.
2. **Judge** (`kind: judge`) is only consulted when every command criterion passes, and only for prose criteria. It sees raw evidence — diff since the last green checkpoint, tool output, changed files — never the executor's summary, and it is told to fail when evidence is insufficient.
3. **Critic** is a separate cheap call that rates *progress*, not completion. It feeds stuck detection and the next plan; it can never mark a mission done.

If a command criterion fails, the judge is skipped entirely — no point paying for prose review of a broken build.

Verification and tool commands run with `PYTHONDONTWRITEBYTECODE=1` and a `PYTHONPYCACHEPREFIX` inside `.deeploop/`, so they never read or write source-tree bytecode caches. Without that, CPython's `(mtime seconds, size)` cache check lets a fast loop judge stale code: an agent edit of the same size within the same second as the previous run reuses the old `.pyc` and the verifier reports a failure that no longer exists.

## Termination and failure modes

- **Stuck detection**: consecutive iterations with no file changes, the same error signature repeating, or the critic reporting no progress. On stuck: `replan` (bounded by `max_replans`), `ask`, or `halt`.
- **Budget exhaustion**: `halt`, `ask` (the human may grant a bounded extension), or `degrade` (switch to `fallback_actor` and continue with an extension).
- **Rollback**: the controller records a green checkpoint each time the verifier passes; `CheckpointManager.rollback(sha)` restores it (`git reset --hard` + `git clean -fd`, so treat the workspace as harness-owned).
- **Crash/Ctrl-C**: state is snapshotted after every iteration; `deeploop resume` picks up iterations, spend, plan, and history from the ledger.

## Repository layout

```
src/deeploop/
  contract.py      task contract: goal, criteria, budget, permissions, escalation, brief
  budget.py        hard limits + live cost accounting
  permissions.py   execution-layer allowlists, path scoping, env scrubbing
  ledger.py        append-only JSONL ledger + resumable state
  controller.py    the loop: plan → act → observe → verify → decide
  verifier.py      verification ladder (command checks, judge) + progress critic
  brief.py         brief folder loading, context manifest, image metadata
  proposal.py      prose → verifiable contract draft, YAML render/validate
  vision.py        optional one-time image descriptions, cached by hash
  prompts.py       role prompts, kept separate from control flow
  llm.py           single choke point for model calls (budget + ledger)
  tools/           shell, files, git; permission-gated
  providers/       OpenAI-compatible client (DeepSeek/OpenRouter/Ollama) + mock
  tui/             Textual UI: mission view, proposal review, Kilo-inspired layout
  cli.py           command line entry point
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The suite covers contract validation, hard budget enforcement, permission bypass attempts, tool behavior, the verification ladder, stuck detection, escalation, resume, brief loading, contract derivation, the CLI, and headless TUI pilot runs (mission view and proposal review).

## Roadmap

- Streaming token output in the TUI (currently whole messages per turn)
- Container-based sandboxing as an alternative to best-effort shell scoping
- Additional providers and a provider-agnostic tool-call adapter
- Cost forecasting per iteration and "would exceed budget" pre-checks
- Multi-mission dashboard reading several ledgers
- Brief diffing: carry the previous contract forward when the brief changes instead of starting over

## License

MIT
