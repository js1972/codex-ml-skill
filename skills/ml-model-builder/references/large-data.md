# Large and Remote Data

## Contents

- [Preflight](#preflight)
- [Panel time-series profiling](#panel-time-series-profiling)
- [Choose the execution location](#choose-the-execution-location)
- [Larger than memory but fits local disk](#larger-than-memory-but-fits-local-disk)
- [Larger than the local machine](#larger-than-the-local-machine)
- [Distributed training](#distributed-training)
- [Sampling](#sampling)
- [Reproducibility and cost](#reproducibility-and-cost)

## Preflight

Before loading data, record:

- compressed and estimated decoded size;
- row/column count or source metadata;
- local available memory and free working disk;
- expected scan count and temporary spill space;
- source format, partitioning, compression and predicate columns;
- network/egress limits and approved compute location;
- privacy/residency constraints;
- time and monetary budget.

CSV can expand several times in memory. Wide strings and categoricals are
especially unpredictable. Do not “try pandas and see” when the estimate is near
the host limit; an out-of-memory kill can lose the entire process.

For remote/object-store inputs, supply `--expected-source-bytes` and an
immutable `--remote-source-version` such as an object version, ETag or snapshot
ID. Supply `--expected-source-rows` when known, but do not use row count alone
to infer scan, disk or memory cost. Fail closed when byte cost or source
identity is unknown. Use `--allow-unknown-remote-preflight` only after explicit
acceptance of bounded scan/cost risk, and record the override. For warehouses,
use dry-run/query-plan estimates and immutable table/snapshot identifiers.
The generic URL profiler records a supplied version as declared but unverified
and marks reproducibility limited because it cannot prove that the scan URL was
bound to that version. Claim exact reproducibility only when a native
object-store/warehouse client verifies the requested version or the input is
content-hashed.

## Panel time-series profiling

Compute full-population series counts, row-count/coverage distributions, date
ranges and gap summaries where the source engine can aggregate them safely.
Do not materialize or plot every series. Bound detailed local diagnostics with
`--max-panel-series` (default 12), select series deterministically, and record
the selection rule. Pull only aggregated profiles and bounded traces from a
remote panel. Run rolling-origin and per-horizon validation separately; the
profiler does not establish forecast validity.

## Choose the execution location

Use the smallest system that safely performs the work:

| Situation | Preferred route |
|---|---|
| Fits comfortably in memory | pandas/in-memory modeling |
| Exceeds memory but fits local disk | DuckDB/Polars streaming with Parquet and disk spill |
| Already in a warehouse/lake | Push profiling/features down to BigQuery, Snowflake, Databricks SQL, Spark, Athena, Trino, or the existing engine |
| Exceeds local disk/CPU or would take excessive scans | Cloud VM/managed cluster close to the data |
| Training algorithm supports out-of-core batches | Incremental/out-of-core estimator over partitioned data |

Avoid downloading a huge warehouse table solely to run local EDA. Move code to
data, return aggregated profiles and bounded samples, and preserve immutable
query/snapshot identifiers.

## Larger than memory but fits local disk

Use `scripts/profile_dataset.py --engine duckdb`; auto mode routes large local
files when its footprint estimate exceeds the memory budget. Configure:

```text
--duckdb-memory-limit 4GB
--duckdb-temp-directory /fast/disk/duckdb-temp
--threads 4
--expected-source-bytes 50000000000
```

Prefer typed, partitioned Parquet over CSV after a validated one-time
conversion. Push column selection, filters and aggregations into DuckDB. Compute
row/missing counts exactly; label approximate cardinalities, quantiles and
duplicate screens as approximate and verify business-critical keys exactly.

Ensure temporary disk can hold spills and intermediate joins. A disk-backed
engine avoids RAM exhaustion but does not make an undersized disk safe.

## Larger than the local machine

Choose from:

- **Warehouse SQL:** best for profiles, joins, temporal coverage, feature
  tables and reproducible samples where the data already lives.
- **Lakehouse/Spark/Dask/Ray:** useful for distributed transformations or
  algorithms that genuinely require multiple workers.
- **Larger single cloud VM:** often simpler and faster than a cluster for
  tabular boosting when the prepared matrix fits one large-memory node.
- **Managed ML services:** useful when training, tracking, permissions and
  deployment are already standardized there.

Do not copy credentials into artifacts. Use the platform's identity mechanism,
read-only access for analysis, bounded queries, partition pruning and cost
controls. Record warehouse project/database, table or snapshot version, query
text/hash, engine version and execution timestamp.

## Distributed training

Distributed compute is not automatically better. Network shuffle and
coordination can dominate. First reduce data with:

- correct row/decision grain;
- unused-column removal based on the prediction contract;
- partition pruning and label-maturity filters;
- safe pre-aggregation;
- compact dtypes and Parquet;
- representative search samples.

For boosting, consider native distributed LightGBM/XGBoost/CatBoost or a larger
single node. For linear models or neural methods, use minibatch/streaming
training when scientifically appropriate. Preserve group/time folds across
partitions and prevent fold-local preprocessors from aggregating globally.

## Sampling

Sampling is valid for:

- plotting;
- model-family/hyperparameter screening;
- expensive explanation prototypes.

Do not sample headline row counts, join integrity, rare-event support, label
maturity, or split feasibility without explicit error bounds. Preserve time,
groups, sources, rare classes and regime coverage. Compare the sample with the
full population using feature-only aggregates, then refit the frozen pipeline
on the intended full development population.

## Reproducibility and cost

Persist:

- source snapshot/table versions and fingerprints where available;
- exact SQL/query plan or transformation code;
- partition/filter rules;
- approximate-aggregate functions and error guarantees;
- sample definition and seed;
- engine, cluster/VM shape, memory, disk and worker counts;
- bytes scanned, elapsed time and monetary cost;
- rejected/failed partitions and retry behavior.

Never claim full-data statistics when they came from a sample. Never claim an
approximate distinct count or quantile is exact.
