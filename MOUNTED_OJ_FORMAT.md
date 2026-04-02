# Mounted OJ Format

`POST /run_oj_cases` judges code against cases stored on a mounted disk instead of sending full testcase payloads through the gateway.

## Data root

The service resolves the mounted data root in this order:

1. `request.data_dir`
2. `OJ_DATA_ROOT` environment variable
3. `config.dataset.oj_data_root`

The data root contains one directory per problem:

```text
<data_root>/
  <problem_id>/
    problem.json
    checker.cpp
    cases/
      1.in
      1.out
      2.in
      2.out
```

`problem_id` is the directory name and must not contain path separators.

## `problem.json`

```json
{
  "problem_id": "sum_two",
  "time_limit_ms": 1000,
  "memory_limit_mb": 256,
  "shared_files": ["testlib.h"],
  "checker": {
    "source": "checker.cpp",
    "files": [],
    "argv": ["input.txt", "output.txt", "answer.txt"]
  },
  "test_cases": [
    {
      "id": "1",
      "input": "cases/1.in",
      "answer": "cases/1.out"
    }
  ]
}
```

## Semantics

- `shared_files`: copied into the temp workdir before compiling solution and checker.
- `checker.source`: checker source relative to the problem directory.
- `checker.files`: extra checker-only files relative to the problem directory.
- `test_cases[*].input` and `test_cases[*].answer`: files relative to the problem directory.
- The endpoint supports `language="cpp"`, `language="java"`, and `language="py3"` (`python` is accepted as an alias for `py3`).
- `cpp` and `java` compile once per request and then run the requested `case_ids`; `py3` performs a one-time syntax check and then executes the script for each requested case.

## Response semantics

Each requested case returns:

- `passed`
- `verdict`: one of `AC`, `WA`, `TLE`, `RE`, `CE`, `CHECKER_CE`, `ERROR`
- `run_result`
- `check_result`

If the solution fails to compile, every requested case is returned with verdict `CE`.
If the checker fails to compile, every requested case is returned with verdict `CHECKER_CE`.
