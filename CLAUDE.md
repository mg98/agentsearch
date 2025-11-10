# AgentSearch

Research repository for expert finding in the Internet of Agents.
We created a dataset based on Google Scholar profiles, each profile represents an agent. We further generated a dataset of questions.
We try to find the best algorithm to match/rank agents based on their capability of answering a given question.

## Coding Standards

- Keep it simple, keep it dry
- No unnecessary comments
- No overly defensive programming: let exceptions and failures occur, otherwise we don't notice deeper lying issues
- No need to use `typing` module as newer Python versions now support `dict`, `tuple`, `list` natively (this project runs 3.10)
- Check out `agentsearch/utils` for global utilities, e.g., `get_torch_device`
- All scripts are implemented in `scripts/` and run with `python -m scripts.script_name`
- When working with CSV, use `pandas`

## Key Files and Directories

We implement all baselines in `agentsearch/baselines`.
