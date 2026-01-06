# IonQ Batching Performance Solution

## Problem Analysis

### Root Cause
The code is **NOT looping** - it's working correctly but **extremely slowly** because:

1. **Sequential IonQ API calls**: `IonQHardwareBackend.execute_batch()` (line 467-471) executes circuits one-by-one sequentially
2. **No parallel execution**: Each circuit waits for the previous one to complete before submitting
3. **High network latency**: Each IonQ API call has ~3 seconds overhead (network + queue time)

### Current Performance
- **Batch size**: 20-32 circuits per batch
- **Time per circuit**: ~3 seconds
- **Time per batch**: 60-96 seconds
- **Batches per epoch**: ~25 (for 1000 samples)
- **Estimated time per epoch**: 25-40 minutes

### Log Evidence
From `/home/yucelz/Downloads/ionq_logs.txt`:
```
WARNING:q_store.backends.ionq_hardware_backend:Executing 20 circuits sequentially on IonQ. Consider batching on simulator backend for cost efficiency.
WARNING:q_store.backends.ionq_hardware_backend:Executing 12 circuits sequentially on IonQ. Consider batching on simulator backend for cost efficiency.
```

## Solution: Parallel IonQ Execution

### Strategy
Replace sequential execution with **parallel async execution** using asyncio/threading:

1. **Submit all circuits in parallel** (non-blocking)
2. **Poll for results concurrently**
3. **Collect results as they complete**

### Implementation Options

#### Option 1: Asyncio with concurrent.futures (Recommended)
Use ThreadPoolExecutor or ProcessPoolExecutor to parallelize IonQ API calls.

**Pros:**
- No API changes needed
- Simple to implement
- Respects IonQ rate limits

**Cons:**
- Still bound by IonQ API limitations

#### Option 2: True Async with asyncio
Convert IonQ backend to fully async using aiohttp or async cirq-ionq.

**Pros:**
- Maximum performance
- Lower resource usage

**Cons:**
- Requires checking if cirq-ionq supports async
- More code changes

#### Option 3: Batch Submission to IonQ
Submit a single "batch job" to IonQ if supported by their API.

**Pros:**
- Optimal performance
- Single API call

**Cons:**
- May not be supported by IonQ simulator
- Different API for different targets

## Implementation Plan

### Step 1: Add Parallel Execution to IonQHardwareBackend

Modify `execute_batch()` method in `src/q_store/backends/ionq_hardware_backend.py`:

```python
def execute_batch(
    self,
    circuits: List[UnifiedCircuit],
    shots: int = 1000,
    parameters: Optional[List[Dict[str, float]]] = None,
    max_workers: int = 10  # NEW: Parallel workers
) -> List[ExecutionResult]:
    """
    Execute multiple circuits on IonQ in parallel.

    Args:
        circuits: List of UnifiedCircuits to execute
        shots: Number of shots per circuit
        parameters: Optional list of parameter dictionaries
        max_workers: Maximum parallel workers (default: 10)

    Returns:
        List of ExecutionResults
    """
    import concurrent.futures

    logger.info(
        f"Executing {len(circuits)} circuits in parallel on IonQ "
        f"(max_workers={max_workers})"
    )

    if parameters is None:
        parameters = [None] * len(circuits)

    # Execute circuits in parallel using ThreadPoolExecutor
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all circuits
        future_to_idx = {
            executor.submit(self.execute, circuit, shots, params): i
            for i, (circuit, params) in enumerate(zip(circuits, parameters))
        }

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results.append((idx, result))
                logger.info(f"Circuit {idx+1}/{len(circuits)} completed")
            except Exception as e:
                logger.error(f"Circuit {idx+1} failed: {e}")
                raise

    # Sort results by original order
    results.sort(key=lambda x: x[0])
    return [r[1] for r in results]
```

### Step 2: Update IonQBackendClientAdapter

Ensure the adapter passes through the parallelization:

```python
async def _execute_in_thread(self, job_id: str):
    """Execute IonQ backend in thread pool with parallel circuit execution."""
    job = self._jobs[job_id]
    job.status = 'running'

    try:
        # Run blocking execute_batch() in thread with parallelization
        # The execute_batch itself will use ThreadPoolExecutor internally
        results = await asyncio.to_thread(
            self.ionq_backend.execute_batch,
            circuits=job.circuits,
            shots=self.shots,
            max_workers=10  # Parallel execution
        )

        # Success
        job.results = results
        job.status = 'completed'
        job.completed_at = time.time()

        elapsed = job.completed_at - job.submitted_at
        logger.info(
            f"Job {job_id} completed successfully "
            f"({len(results)} results, {elapsed:.1f}s)"
        )

    except Exception as e:
        job.error = str(e)
        job.status = 'failed'
        job.completed_at = time.time()
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
```

### Step 3: Add Rate Limiting (Optional)

To respect IonQ API rate limits:

```python
import time
from threading import Lock

class RateLimiter:
    """Token bucket rate limiter for IonQ API."""

    def __init__(self, max_requests_per_second: int = 10):
        self.max_requests = max_requests_per_second
        self.tokens = max_requests_per_second
        self.last_update = time.time()
        self.lock = Lock()

    def acquire(self):
        """Acquire a token (blocks if necessary)."""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update

            # Refill tokens
            self.tokens = min(
                self.max_requests,
                self.tokens + elapsed * self.max_requests
            )
            self.last_update = now

            # Wait if no tokens available
            if self.tokens < 1:
                sleep_time = (1 - self.tokens) / self.max_requests
                time.sleep(sleep_time)
                self.tokens = 0
            else:
                self.tokens -= 1
```

## Expected Performance Improvement

### Before (Sequential)
- **Time per batch**: 60-96 seconds
- **Time per epoch**: 25-40 minutes
- **Throughput**: ~0.3-0.5 circuits/second

### After (Parallel with 10 workers)
- **Time per batch**: 6-10 seconds (10x improvement)
- **Time per epoch**: 2.5-4 minutes (10x improvement)
- **Throughput**: ~3-5 circuits/second

### Assumptions
- IonQ simulator can handle 10 concurrent requests
- Network latency remains constant
- No API rate limiting issues

## Testing Strategy

1. **Unit Test**: Test parallel execution with mock backend
2. **Integration Test**: Test with IonQ simulator (small batch)
3. **Performance Test**: Measure speedup with different worker counts
4. **Full Test**: Run quick-test mode with real IonQ simulator

## Rollout Plan

1. ✅ **Analysis**: Understand root cause (DONE)
2. **Implementation**: Add parallel execution to IonQHardwareBackend
3. **Testing**: Verify with small dataset
4. **Optimization**: Tune max_workers parameter
5. **Documentation**: Update examples and guides

## Alternative Solutions

### If IonQ API has strict rate limits:
1. Use smaller `max_workers` (e.g., 5 instead of 10)
2. Implement exponential backoff
3. Add retry logic for rate-limited requests

### If parallel execution doesn't help:
1. **Cache quantum circuit results** (already implemented in AsyncQuantumExecutor)
2. **Reduce batch size** (trade-off: more batches but less memory)
3. **Use mock backend for development** (--no-mock flag off)

## Notes

- The async executor already supports batching and caching
- The bottleneck is specifically in `IonQHardwareBackend.execute_batch()`
- The solution maintains backward compatibility
- No changes needed to quantum layers or model code
