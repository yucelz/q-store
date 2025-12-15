#!/usr/bin/env python3
"""
Quick demo showing the logging and benchmarking in action
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from q_store_examples.utils import ExampleLogger


async def demo():
    """Demonstrate the logging and benchmarking system"""
    
    # Initialize logger
    logger = ExampleLogger(
        log_dir="LOG",
        base_dir=os.path.dirname(__file__),
        example_name="demo"
    )
    
    print("\n" + "="*70)
    print("Logging and Benchmarking System Demo")
    print("="*70 + "\n")
    
    # Simulate Example 1
    logger.log_info("Running Example 1: Data Preparation")
    with logger.benchmark_step("Example 1: Data Preparation", 
                                metadata={"samples": 100}):
        await asyncio.sleep(0.05)
        logger.log_info("Generated 100 training samples")
    
    # Simulate Example 2
    logger.log_info("Running Example 2: Model Training")
    step2 = logger.start_step("Example 2: Model Training", 
                              metadata={"epochs": 5, "batch_size": 10})
    await asyncio.sleep(0.1)
    logger.log_info("Training completed")
    logger.end_step(step2, status="completed", 
                    result={"final_loss": 0.234, "accuracy": 0.89})
    
    # Simulate Example 3
    logger.log_info("Running Example 3: Evaluation")
    with logger.benchmark_step("Example 3: Evaluation"):
        await asyncio.sleep(0.03)
        logger.log_info("Model evaluated on test set")
    
    # Simulate an error case
    logger.log_info("Running Example 4: Error Handling Demo")
    step4 = logger.start_step("Example 4: Error Handling Demo")
    try:
        # Simulate some work that might fail
        await asyncio.sleep(0.02)
        logger.log_warning("This is a simulated warning")
        # Don't actually raise error for demo
        logger.end_step(step4, status="completed")
    except Exception as e:
        logger.log_error(f"Example failed: {e}", exc_info=True)
        logger.end_step(step4, status="failed", result={"error": str(e)})
    
    # Finalize
    logger.finalize()
    logger.print_summary()
    
    print("\n" + "="*70)
    print("Files Created:")
    print("="*70)
    print(f"📄 Log File:       {logger.log_file}")
    print(f"📊 Benchmark JSON: {logger.benchmark_file}")
    print("="*70 + "\n")
    
    # Show file sizes
    log_size = logger.log_file.stat().st_size
    benchmark_size = logger.benchmark_file.stat().st_size
    print(f"✓ Log file:       {log_size:,} bytes")
    print(f"✓ Benchmark file: {benchmark_size:,} bytes")
    print(f"✓ Total steps:    {len(logger.benchmarks['steps'])}")
    print(f"✓ Total time:     {logger.benchmarks['total_duration_ms']:.2f}ms\n")


if __name__ == "__main__":
    asyncio.run(demo())
