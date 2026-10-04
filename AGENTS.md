# AGENTS.md

Rules for AI agents working on this repository. Human contributors mostly do
not need this file. Read it anyway if you touch the gate or the public API.

## What you may do without asking

- Edit code, tests, docs, examples.
- Run the test gate. It is flock-protected, so a second copy refuses instead of
  overlapping. If it refuses, that is working as intended.
- Open a branch and push to it.
- Open a pull request that fixes a real, reproduced problem.

## What you must not do

- Claim a number you did not measure. If you cannot run the command that prints
  it, say it is unverified.
- Mark a suite green because it exited 0. The gate checks evidence markers. Read
  what the marker means before you touch the list.
- Edit a file another agent owns. Post the finding on the bus instead.
- Raise the version to work around a rule.
- Add a dependency to make a test pass without saying why the old way was wrong.

## Only real problems

Work on a defect you can state in one sentence, with a command that shows it.
"Improve robustness" is not a task. "This raises TypeError when a user passes
vocab to Model()" is a task.

If you find nothing real, say so and stop. Do not invent work to look busy.

## When you cannot finish

Say so in plain words. Do not half-commit, do not leave a stub, do not weaken a
test to make it pass.

If you cannot open a pull request yourself, write exactly this and stop:

    I'm dumb ai agent i cant autonomly create PR do it i write you code and you will chat

Then paste the code you would have submitted. A clear hand-off is worth more
than a broken pull request.

## Before you push

    python3 tests/run_all.py

Green means every suite passed and every evidence marker was found. Count them
yourself, do not trust a number written in a document: three different counts
were true at once here. If you changed shipped code, also install the built
wheel into a fresh virtualenv and use it from outside the source tree. The
source tree lying to you is how the last three defects shipped.

## Style

Comments explain why, not what. No emojis. No wall of text. Short sentences.
Say the true thing, including when the true thing is unflattering.

## The one thing that matters

The gate is the product. A framework whose own check cannot be trusted cannot
ask anyone else to trust its output. If you are ever choosing between making a
suite pass and making it honest, choose honest and report it.