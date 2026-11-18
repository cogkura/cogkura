# Retrieval profiling

The 0.20.0 investigation uses standalone scripts. The ordinary Core APIs have no
new profiling switch or persistent instrumentation. Run measurements in fresh
processes and keep timing runs separate from diagnostics:

```bash
uv run python scripts/profile_retrieval.py --backend memory --observations 100 \
  --revisions 5 --recalls 5 --uninstrumented --output memory.json
uv run python scripts/profile_retrieval.py --backend sqlite --observations 100 \
  --revisions 5 --stage-diagnostics --context --output stages.json
uv run python scripts/profile_retrieval.py --observations 100 --revisions 5 \
  --allocations --output allocations.json
uv run python scripts/profile_retrieval.py --backend sqlite --observations 100 \
  --revisions 5 --sql-trace --output sql.json
uv run python scripts/profile_fresh_process.py --observations 100 --revisions 5 \
  --processes 3 --output restart.json
uv run python scripts/run_software_project_workload.py --backend sqlite --projects 5 \
  --heterogeneous --relationships-per-project 5 --output projects.json
```

The mixed fixture is unchanged from the 0.19.6 workload. `--recalls` uses a fixed
sequence of distinct queries and records each observation; it is not a set of
independent devices or hosts. Stage, allocation and SQL modes are mutually
exclusive and increase runtime. Stage timers are nested, may overlap during
`asyncio.gather`, and must not be added as an exact end-to-end breakdown.
`returned_items` counts collection elements returned to Python, not distinct
database rows. The cProfile default captures function calls and inclusive and
exclusive time; the allocation mode captures traced peak and net retained blocks,
not all allocation events. SQL tracing counts executed SQLite statements; it
does not measure bytes read or physical I/O.

For PostgreSQL, install the optional extra and pass `--backend postgres
--postgres-url URL`. Use a disposable database: the script applies migrations
and clears the tenant named `benchmark` before and after the run. PostgreSQL
SQL tracing is not provided by the SQLite tracing option.

The fresh-process runner builds and closes a SQLite fixture, starts new Python
processes and checks both first and subsequent recall outputs against it. The
OS page cache is not cleared, so fresh process does not mean cold device. Peak
process RSS includes fixture construction and uses platform-specific units.

The heterogeneous project fixture varies incident language and graph density,
but remains synthetic and is not evidence of representative production Agent
performance. Semantic-history workloads can surface the documented 0.19.6
withdrawal/knowledge-time error and return a failing process status rather than
silently hiding it. See the [design](design-retrieval-performance-0.20.0.md)
and [release preparation](findings/0.20.0-release-preparation.md) for the scope,
evidence and outstanding decisions.
