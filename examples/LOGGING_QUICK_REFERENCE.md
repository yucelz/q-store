# Quick Reference: Logging & Benchmarking System

## Overview
When `examples_v3_2.py` (or any example using the utilities) runs, it automatically:
1. ✅ Creates a `LOG/` folder
2. ✅ Creates a timestamped log file: `log-YYYYMMDD_HHMMSS.txt`
3. ✅ Creates a benchmark JSON file: `benchmark-<example_name>-YYYYMMDD_HHMMSS.json`

## Quick Start

### For New Examples
```python
from q_store_examples.utils import ExampleLogger

# 1. Initialize
logger = ExampleLogger(
    log_dir="LOG",
    base_dir="/path/to/examples",
    example_name="my_example"
)

# 2. Log messages
logger.log_info("Starting...")
logger.log_debug("Detailed info...")
logger.log_warning("Warning message")
logger.log_error("Error occurred", exc_info=True)

# 3. Track performance with context manager (recommended)
with logger.benchmark_step("Data Processing"):
    # Your code here
    pass

# 4. Or manually
step_idx = logger.start_step("Training", metadata={"epochs": 10})
# ... do work ...
logger.end_step(step_idx, status="completed", result={"loss": 0.123})

# 5. Finalize
logger.finalize()
logger.print_summary()
```

## Files Created

### LOG Directory Structure
```
examples/
└── LOG/
    ├── log-20251215_214109.txt                      # Detailed text logs
    ├── benchmark-demo-20251215_214109.json          # Performance data
    ├── benchmark-examples_v3_2-20251215_213644.json # Multiple runs
    └── ...
```

### Log File Content
- Timestamped entries
- All log levels (DEBUG, INFO, WARNING, ERROR)
- Stack traces for errors
- Example: `2025-12-15 21:41:09 - demo - INFO - Generated 100 training samples`

### Benchmark JSON Content
```json
{
  "example_name": "demo",
  "start_time": "2025-12-15T21:41:09.950840",
  "end_time": "2025-12-15T21:41:10.162659",
  "total_duration_ms": 210.80,
  "steps": [
    {
      "name": "Example 1: Data Preparation",
      "duration_ms": 50.79,
      "status": "completed",
      "metadata": {"samples": 100},
      "result": {}
    }
  ]
}
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Auto Directory** | LOG/ created automatically if missing |
| **Timestamps** | All files timestamped (no overwrites) |
| **Dual Output** | Console (INFO+) + File (DEBUG+) |
| **Rich Metadata** | Store custom data with each step |
| **Error Tracking** | Automatic error logging with traces |
| **JSON Export** | Easy data analysis and comparison |
| **Thread Safe** | Works with async operations |
| **Optional** | Examples can use it or not |

## Testing

```bash
# Quick test
cd examples
python test_logging.py

# Demo
python demo_logging.py

# Run actual examples with logging
python src/q_store_examples/examples_v3_2.py
```

## Analyzing Results

```python
from q_store_examples.utils import BenchmarkCollector

collector = BenchmarkCollector(log_dir="LOG")
collector.load_all_benchmarks()
collector.compare_examples(example_name="examples_v3_2")
```

## Common Patterns

### Pattern 1: Simple Example
```python
logger = ExampleLogger(example_name="simple")
with logger.benchmark_step("Main Task"):
    # Do work
    pass
logger.finalize()
```

### Pattern 2: Multiple Steps
```python
logger = ExampleLogger(example_name="multi_step")

with logger.benchmark_step("Step 1: Setup"):
    setup_data()

with logger.benchmark_step("Step 2: Process"):
    process_data()

with logger.benchmark_step("Step 3: Save"):
    save_results()

logger.finalize()
logger.print_summary()
```

### Pattern 3: Error Handling
```python
logger = ExampleLogger(example_name="robust")

step = logger.start_step("Critical Operation")
try:
    risky_operation()
    logger.end_step(step, status="completed")
except Exception as e:
    logger.log_error(f"Failed: {e}", exc_info=True)
    logger.end_step(step, status="failed", result={"error": str(e)})
    raise

logger.finalize()
```

## Tips

1. **Use context managers** (`with logger.benchmark_step(...)`) for cleaner code
2. **Add metadata** to steps for better analysis later
3. **Store results** in the step for tracking metrics
4. **Always call finalize()** to save final benchmark data
5. **Use descriptive step names** for clarity in reports

## Documentation
- Full API: [README_UTILS.md](src/q_store_examples/README_UTILS.md)
- Implementation: [LOGGING_IMPLEMENTATION_SUMMARY.md](LOGGING_IMPLEMENTATION_SUMMARY.md)
