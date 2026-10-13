# Frozen CPU references

The maintainers generated these frozen values from the JetSpec++ reference
implementation. Regeneration is not part of this repository.

The ordinary CPU suite reads `fixtures/golden/*.json` and `*.safetensors` through
`golden_values.py` without importing the reference implementation. Each JSON
file is indexed by test name, parameter IDs, label and repeated-call index.
Tensor values are stored without conversion or rounding; loading clones them
so tests cannot mutate another case's values. The original exact comparisons
and tolerance assertions remain in the tests.

The public-output mutation checks are available with:

```sh
python tests/jetspec_plus_plus/golden/mutate.py --report /tmp/mutations.json
```

The check changes one returned value per comparison file and succeeds only
when every selected test rejects that change with an assertion.
