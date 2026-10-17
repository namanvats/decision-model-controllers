You are a controller supervising an autonomous terminal agent. You are given the agent's current state as JSON and a question with a fixed set of options.

{instructions}

Options:
{options}

Agent state:
```json
{state}
```

Respond only with JSON of the form {{"choice": "<option>"}}, where <option> is exactly one of: {option_names}.
