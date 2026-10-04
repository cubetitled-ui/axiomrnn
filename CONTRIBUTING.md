# Contributing

Patches are welcome, including from agents. Read [AGENTS.md](AGENTS.md) first if
you are an automated contributor.

## The rule that matters

Every number in this project must come from a command you can paste. If you
cannot run it, mark it unverified or leave it out. A wrong number costs more
than a missing one, because a reader cannot tell the difference without
reproducing your work.

## Setup

    python3 -m venv .venv
    . .venv/bin/activate
    pip install -e ".[dev]"

TensorFlow is a hard dependency. It is a large download. If you only want the
memory planner, `axplan` and `axon` import with numpy alone.

## Run the tests

    python3 tests/run_all.py

About twenty minutes and roughly 600 MB of TensorFlow. It runs every example in
`examples/` as a suite, plus the unit tests and the C++ kernels. Count them
yourself rather than trusting a number here. The gate holds a lock, so a second
copy exits instead of overlapping.

To check one thing quickly:

    python3 tests/test_credit.py

## What a green gate means

Every suite passed and every declared evidence marker was found. A suite with
no markers passes on exit code alone, and the gate prints a line saying so. That
line is not decoration. If you add an example that only checks it did not crash,
leave the line visible.

## Style

Follow the surrounding code. English, short sentences, comments that say why.
No emojis.

## Pull requests

One problem per pull request. Say what was broken, show the command that shows
it broken, show the command that shows it fixed.

If your change makes the gate slower, say by how much and why the cost is worth
it. A suite that adds ten minutes needs a reason.

## Commit messages

Plain sentences. What changed and why. Reference the issue if there is one.

## Adding an example

Put it in `examples/`, add it to the SUITES table in `tests/run_all.py`, and give
it at least one evidence marker. An example that is not in the gate will rot,
and rot silently.

If it is slow, give it an environment variable that reduces the training and use
that variable in the gate. Keep the code path identical; shrink only the work.

## Licence

Apache-2.0. By contributing you agree your contribution is published under it.