# Contributing

Thanks for considering contributing to Lok Sabha Debates Explorer. This is a civic-tech project
built entirely on local/self-hosted models (no paid LLM API, by explicit design), so setup takes
a bit more than `pip install` — see the [README](README.md#running-it) for the full local setup.

## Before you start

- Read `PROGRESS.md` first. It's the running decision/reasoning log for this project — why
  things are built the way they are, what's already been tried, and what's still open. It will
  save you from re-discovering something already fixed (or already ruled out) here.
- `CLAUDE.md` describes the current architecture as it stands.
- The README's "Known limitations" section and `PROGRESS.md`'s ranked issue notes are good
  starting points if you're looking for something concrete to work on.

## Workflow

- `main` is protected — changes land via pull request, not direct pushes.
- Fork the repo, create a branch off `main`, make your change, and open a PR against `main`.
- Keep PRs focused: one logical change per PR is much easier to review than a bundle of
  unrelated fixes.
- Write commit messages and PR descriptions that explain *why*, not just *what* — especially for
  anything touching the pipeline (`pipeline/`), where a fix is often motivated by a specific,
  measured bug (garbled OCR text, a mis-segmented speech boundary, a truncated model input,
  etc.). If you found and measured a real bug, say so and how you confirmed it — that context is
  as valuable as the fix itself for a project like this.
- Add tests where practical.  For pipeline-stage changes, "practical" often means a small script
  or note in the PR showing the fix against real (or representative) data, since much of this
  project's reliability work has come from measuring against real data rather than unit tests
  alone — see `PROGRESS.md` for many examples of that practice.

## Reporting issues

Open a GitHub issue. For data-quality issues (a wrong translation, a misattributed speech, a bad
geocode match, etc.), include the `speech_id` or a snippet of the source text if you can — that
makes it possible to reproduce and measure, which is the standard this project holds fixes to.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By participating, you're
expected to uphold it.
