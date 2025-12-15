# Q-Store Examples Utilities

## Overview

The `utils.py` module provides comprehensive logging and benchmarking functionality for Q-Store examples. All example scripts can optionally use these utilities to track execution and performance.

## Features

### 1. **ExampleLogger** - Main Logging and Benchmarking Class

#### Automatic Directory and File Creation
- Creates `LOG/` directory automatically
- Generates timestamped log files: `log-YYYYMMDD_HHMMSS.txt`
- Generates timestamped benchmark files: `benchmark-<example_name>-YYYYMMDD_HHMMSS.json`

#### Dual Logging
- Console output (INFO level and above)
- File output (DEBUG level and above)
- Timestamped entries with proper formatting

#### Benchmark Tracking
- Records start and end times for all steps
- Calculates duration in milliseconds
- Stores metadata and results for each step
- Exports to JSON for easy analysis

## Usage

### Basic Example

```python
from q_store_examples.utils import ExampleLogger

# Initialize logger
logger = ExampleLogger(
    log_dir="LOG",                    # Directory for logs (default: "LOG")
    base_dir="/path/to/examples",     # Base directory (default: current dir)
    example_name="my_example"         # Example name for file naming
)

# Log messages
logger.log_info("Starting example")
logger.log_debug("Detailed debug information")
logger.log_warning("Something to note")
logger.log_error("An error occurred", exc_info=True)

# Track benchmark steps manually
step_idx = logger.start_step("Training", metadata={"epochs": 10})
# ... do work ...
logger.end_step(step_idx, status="completed", result={"loss": 0.123})

# Or use context manager (recommended)
with logger.benchmark_step("Data Processing"):
    # ... do work ...
    pass

# Finalize and save
logger.finalize()
logger.print_summary()
```

### Integration with Async Functions

```python
import asyncio
from q_store_examples.utils import ExampleLogger

async def train_model():
    logger = ExampleLogger(example_name="training_example")
    
    with logger.benchmark_step("Model Setup"):
        # Setup code
        await asyncio.sleep(0.1)
    
    with logger.benchmark_step("Training Loop"):
        # Training code
        await asyncio.sleep(1.0)
    
    logger.finalize()
    logger.print_summary()

asyncio.run(train_model())
```

## File Structure

### LOG Directory
```
examples/
├── LOG/
│   ├── log-20251215_213644.txt
│   ├── log-20251215_213914.txt
│   ├── benchmark-examples_v3_2-20251215_213644.json
│   └── benchmark-test_example-20251215_213914.json
└── src/
    └── q_store_examples/
        ├── utils.py
        └── examples_v3_2.py
```

### Log File Format
```
2025-12-15 21:39:14 - q_store_examples.test_example.20251215_213914 - INFO - ExampleLogger initialized for 'test_example'
2025-12-15 21:39:14 - q_store_examples.test_example.20251215_213914 - INFO - Started step: Step 1: Setup
2025-12-15 21:39:15 - q_store_examples.test_example.20251215_213914 - INFO - Ended step: Step 1: Setup - Status: completed - Duration: 100.31ms
```

### Benchmark JSON Format
```json
{
  "example_name": "test_example",
  "start_time": "2025-12-15T21:39:14.914285",
  "end_time": "2025-12-15T21:39:15.087497",
  "total_duration_ms": 171.966796875,
  "steps": [
    {
      "name": "Step 1: Setup",
      "start_time": "2025-12-15T21:39:14.914980",
      "start_time_ms": 1765834754914.9893,
      "end_time": "2025-12-15T21:39:15.015299",
      "end_time_ms": 1765834755015.296,
      "duration_ms": 100.306640625,
      "status": "completed",
      "metadata": {
        "task": "initialization"
      },
      "result": {
        "items_created": 5
      }
    }
  ]
}
```

## API Reference

### ExampleLogger

#### Constructor
```python
ExampleLogger(
    log_dir: str = "LOG",
    base_dir: Optional[str] = None,
    example_name: str = "example"
)
```

#### Logging Methods
- `log_info(message: str)` - Log info message
- `log_debug(message: str)` - Log debug message
- `log_warning(message: str)` - Log warning message
- `log_error(message: str, exc_info: bool = False)` - Log error message

#### Benchmarking Methods
- `start_step(step_name: str, metadata: Optional[Dict] = None) -> int` - Start timing a step
- `end_step(step_index: Optional[int] = None, status: str = "completed", result: Optional[Dict] = None)` - End timing a step
- `benchmark_step(step_name: str, metadata: Optional[Dict] = None)` - Context manager for timing

#### Finalization
- `finalize()` - Save final benchmark data
- `print_summary()` - Print formatted summary
- `get_step_summary() -> List[Dict]` - Get step summary data

### BenchmarkCollector

For comparing multiple benchmark runs:

```python
from q_store_examples.utils import BenchmarkCollector

collector = BenchmarkCollector(log_dir="LOG")
collector.load_all_benchmarks()
collector.compare_examples(example_name="examples_v3_2")
```

## Examples

See `examples_v3_2.py` for a complete integration example showing:
- Global logger initialization
- Step tracking for each example function
- Error handling with logging
- Final summary generation

Run the test script to verify functionality:
```bash
cd examples
python test_logging.py
```

## Benefits

1. **Consistent Logging**: All examples use the same logging format
2. **Performance Tracking**: Automatic benchmarking of all operations
3. **Debugging**: Detailed timestamped logs for troubleshooting
4. **Analysis**: JSON benchmarks for performance analysis and comparison
5. **Optional**: Examples can choose to use or not use the utilities
6. **Thread-Safe**: Proper logging configuration for async operations
