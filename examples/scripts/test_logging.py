#!/usr/bin/env python3
"""
Quick test script to verify the logging and benchmarking system
"""

import asyncio
import time
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from q_store_examples.utils import ExampleLogger


async def test_logging():
    """Test the ExampleLogger functionality"""
    print("Testing ExampleLogger...")
    
    # Create logger
    logger = ExampleLogger(
        log_dir="LOG",
        base_dir=os.path.dirname(__file__),
        example_name="test_example"
    )
    
    # Test step tracking
    logger.log_info("Starting test example")
    
    # Step 1
    step1_idx = logger.start_step("Step 1: Setup", metadata={"task": "initialization"})
    await asyncio.sleep(0.1)  # Simulate work
    logger.end_step(step1_idx, status="completed", result={"items_created": 5})
    
    # Step 2 with context manager
    with logger.benchmark_step("Step 2: Processing"):
        await asyncio.sleep(0.05)
        logger.log_info("Processing data...")
    
    # Step 3
    step3_idx = logger.start_step("Step 3: Finalization")
    await asyncio.sleep(0.02)
    logger.end_step(step3_idx, status="completed")
    
    # Finalize
    logger.finalize()
    logger.print_summary()
    
    print(f"\n✓ Log file created: {logger.log_file}")
    print(f"✓ Benchmark file created: {logger.benchmark_file}")
    
    # Verify files exist
    if logger.log_file.exists():
        print(f"✓ Log file exists and has {logger.log_file.stat().st_size} bytes")
    else:
        print("✗ Log file NOT found!")
    
    if logger.benchmark_file.exists():
        print(f"✓ Benchmark JSON exists and has {logger.benchmark_file.stat().st_size} bytes")
        
        # Show benchmark content
        import json
        with open(logger.benchmark_file) as f:
            data = json.load(f)
            print(f"\nBenchmark data:")
            print(f"  Example: {data['example_name']}")
            print(f"  Steps: {len(data['steps'])}")
            print(f"  Total duration: {data['total_duration_ms']:.2f}ms")
    else:
        print("✗ Benchmark JSON NOT found!")
    
    return logger


if __name__ == "__main__":
    logger = asyncio.run(test_logging())
    print("\n✓ Test completed successfully!")
