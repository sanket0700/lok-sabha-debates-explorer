---
name: Bug report
about: Something's wrong with the pipeline, data, or app
title: ''
labels: bug
assignees: ''
---

**What's wrong**
A clear description of the bug.

**Where**
Which part: scrape / extract / segment / translate / NER / geocode / topic / sentiment / embed /
aggregate / app (Explore) / app (Insights) / other.

**Reproduce it**
- `speech_id` or `sitting_id` if this is a data-quality issue (a wrong translation, a
  misattributed speech, a bad geocode match, etc.) — this makes it possible to reproduce and
  measure, which is the standard this project holds fixes to (see `PROGRESS.md`).
- Command(s) run, if a pipeline stage.
- URL/steps, if the app.

**Expected vs. actual**
What you expected to happen vs. what actually happened.

**Environment**
- OS / Python version
- Local or Docker setup
