# Q-Store Performance Benchmarks

Comprehensive benchmark suite for tracking Q-Store performance and detecting regressions across releases.

## Overview

The benchmark suite (`test_benchmarks.py`) provides performance measurements for:

1. **Circuit Creation** - Creating various quantum circuits
2. **Profiling Operations** - Profiling circuit performance
3. **Verification Operations** - Checking circuit properties
4. **Visualization Operations** - Rendering circuits
5. **Tomography Operations** - State reconstruction
6. **End-to-End Workflows** - Complete operation chains
7. **Scaling Tests** - Performance vs circuit size
8. **Regression Baselines** - Reference metrics for regression testing

## Running Benchmarks

### Run All Benchmarks

```bash
pytest tests/test_benchmarks.py -v -m benchmark -s
```

### Run Specific Benchmark Categories

```bash
# Circuit creation benchmarks
pytest tests/test_benchmarks.py::TestCircuitCreationBenchmarks -v -s

# Profiling benchmarks
pytest tests/test_benchmarks.py::TestProfilingBenchmarks -v -s

# Verification benchmarks
pytest tests/test_benchmarks.py::TestVerificationBenchmarks -v -s

# Workflow benchmarks
pytest tests/test_benchmarks.py::TestWorkflowBenchmarks -v -s

# Scaling benchmarks
pytest tests/test_benchmarks.py::TestScalingBenchmarks -v -s

# Regression baselines
pytest tests/test_benchmarks.py::TestRegressionBaselines -v -s
```

### Run Without Benchmark Marker Filter

```bash
pytest tests/test_benchmarks.py -v --tb=short
```

## Benchmark Categories

### 1. Circuit Creation Benchmarks

Measure time to create quantum circuits of various sizes:

- **Bell State**: 2-qubit entangled state (~0.01ms)
- **GHZ State**: 3-qubit entangled state (~0.01ms)
- **Parameterized Circuit**: Circuit with rotation gates (~0.01ms)
- **Large Circuit**: 10 qubits, 50 gates (~0.2ms)

### 2. Profiling Benchmarks

Measure circuit profiling performance:

- **Small Circuit**: 2 qubits, 2 gates (~0.1ms)
- **Medium Circuit**: 5 qubits, 20 gates (~1ms)
- **Batch Profiling**: 10 circuits (~3ms)

### 3. Verification Benchmarks

Measure verification operation performance:

- **Unitarity Check**: Verify circuit is unitary (~0.5ms)
- **Circuit Equivalence**: Check two circuits are equivalent (~0.3ms)

### 4. Visualization Benchmarks

Measure visualization rendering time:

- **Small Circuit**: 2 qubits (~0.005ms)
- **Medium Circuit**: 4 qubits (~0.01ms)

### 5. Tomography Benchmarks

Measure state/process reconstruction:

- **State Reconstruction**: Reconstruct state from measurements (~0.01ms)

### 6. End-to-End Workflows

Measure complete operation workflows:

- **Create → Profile → Visualize**: Full workflow (~0.2ms)
- **Create → Verify → Profile**: Verification workflow (~0.5ms)
- **Batch Processing**: 10 circuits (~4ms)

### 7. Scaling Benchmarks

Test performance scaling with circuit size:

- **Qubit Count Scaling**: 2, 4, 6, 8, 10 qubits
- **Gate Count Scaling**: 10, 20, 50, 100 gates

Expected: roughly linear or better scaling

### 8. Regression Baselines

Establish baseline metrics for regression testing:

- **Bell State Workflow**: < 0.2ms (baseline)
- **Verification Performance**: < 1ms (baseline)
- **Profiling Performance**: < 0.5ms (baseline)

## Performance Targets

### Current Performance Baselines (v4.0)

| Operation | Target | Measured |
|-----------|--------|----------|
| Bell state creation | < 0.01ms | ~0.008ms |
| Small circuit profile | < 0.2ms | ~0.124ms |
| Circuit visualization | < 0.01ms | ~0.005ms |
| Unitarity verification | < 1ms | ~0.5ms |
| Complete workflow | < 0.5ms | ~0.2ms |

## Interpreting Results

### Benchmark Output Format

```
test_name: X.XXXms/iter (N iters)
```

- **X.XXXms/iter**: Average time per iteration in milliseconds
- **(N iters)**: Number of iterations performed

### Performance Guidelines

- **Excellent**: < 1ms for individual operations
- **Good**: 1-10ms for complex operations
- **Acceptable**: 10-100ms for batch operations
- **Needs Optimization**: > 100ms for simple operations

### Scaling Expectations

- **Qubit scaling**: Should be roughly O(n²) or better
- **Gate scaling**: Should be roughly O(n) or better
- **10x size increase**: Should not be >15x slower

## Regression Testing

### Establishing Baselines

1. Run benchmarks on stable release:
   ```bash
   pytest tests/test_benchmarks.py::TestRegressionBaselines -v -s > baseline_v4.0.txt
   ```

2. Save baseline results for comparison

### Detecting Regressions

1. Run benchmarks after changes:
   ```bash
   pytest tests/test_benchmarks.py -v -s > current_results.txt
   ```

2. Compare with baseline:
   - Performance degradation >20%: Investigate
   - Performance degradation >50%: Likely regression
   - Performance improvement: Good! Verify correctness

### CI/CD Integration

Add to GitHub Actions workflow:

```yaml
- name: Run Performance Benchmarks
  run: |
    pytest tests/test_benchmarks.py -v -m benchmark --tb=short
    
- name: Check for Performance Regressions
  run: |
    # Compare with baseline (implement custom script)
    python scripts/check_performance_regression.py
```

## Adding New Benchmarks

### Benchmark Template

```python
@pytest.mark.benchmark
class TestMyBenchmarks:
    """Benchmark description."""
    
    def test_benchmark_my_operation(self, benchmark_iterations=100):
        """Benchmark description."""
        # Setup
        setup_data = create_test_data()
        
        def operation():
            # Operation to benchmark
            result = my_operation(setup_data)
            return result
        
        # Run benchmark
        result = benchmark(operation, benchmark_iterations)
        
        # Assert performance target
        assert result.avg_time < 0.01  # 10ms target
        
        # Print result
        print(f"\n{result}")
```

### Best Practices

1. **Isolate Setup**: Don't include setup time in benchmarks
2. **Multiple Iterations**: Use enough iterations for stable averages
3. **Realistic Workloads**: Benchmark real-world usage patterns
4. **Set Targets**: Include performance assertions
5. **Print Results**: Use `-s` flag to see timing output
6. **Document Expectations**: Explain what good performance looks like

## Profiling for Optimization

### Using Python Profiler

```python
import cProfile
import pstats

# Profile a specific test
cProfile.run('test_function()', 'profile_stats')
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 slowest functions
```

### Using py-spy for Live Profiling

```bash
# Install py-spy
pip install py-spy

# Profile benchmark run
py-spy record -o profile.svg -- pytest tests/test_benchmarks.py::test_name -s

# View flamegraph
firefox profile.svg
```

## Continuous Monitoring

### Track Performance Over Time

1. **Store Results**: Save benchmark results with git commits
2. **Plot Trends**: Create performance graphs over time
3. **Set Alerts**: Alert on significant regressions
4. **Review Regularly**: Check benchmarks in code reviews

### Example Monitoring Script

```bash
#!/bin/bash
# Save benchmark results with git hash
HASH=$(git rev-parse --short HEAD)
pytest tests/test_benchmarks.py -v -s > "benchmarks/results_${HASH}.txt"
```

## Troubleshooting

### Benchmarks Taking Too Long

- Reduce iteration counts in test parameters
- Run specific test categories instead of all benchmarks
- Use pytest's `-k` flag to run specific tests

### Inconsistent Results

- Ensure system is not under heavy load
- Run multiple times and average results
- Check for background processes
- Consider using `pytest-benchmark` plugin for more stable measurements

### Failed Assertions

- Check if performance targets are too aggressive
- Verify system specifications match development environment
- Investigate potential regressions in dependencies

## Related Documentation

- [Testing Guide](../README.md#testing)
- [Integration Tests](./test_integration_v4.py)
- [Profiling Module](../src/q_store/profiling/)
- [Verification Module](../src/q_store/verification/)

## Contributing

When adding new features:

1. Add corresponding benchmarks
2. Set reasonable performance targets
3. Document expected performance characteristics
4. Run benchmarks before and after changes
5. Include performance notes in PR description
