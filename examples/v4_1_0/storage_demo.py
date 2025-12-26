"""
Phase 3 Storage Demo - Production-Grade Async Storage

Demonstrates all Phase 3 storage components:
1. AsyncBuffer - Non-blocking ring buffer
2. AsyncMetricsWriter - Background Parquet writer
3. CheckpointManager - Zarr checkpointing
4. AsyncMetricsLogger - High-level logging API

Run this to validate Phase 3 implementation!
"""

import asyncio
import numpy as np
from pathlib import Path
import time
import shutil

from q_store.storage import (
    AsyncBuffer,
    AsyncMetricsWriter,
    CheckpointManager,
    TrainingMetrics,
    AsyncMetricsLogger,
)


# ============================================================================
# Demo 1: AsyncBuffer - Never Blocks!
# ============================================================================

async def demo_async_buffer():
    """Demo non-blocking buffer with overflow handling."""
    print("\n" + "="*70)
    print("DEMO 1: AsyncBuffer - Non-Blocking Ring Buffer")
    print("="*70)
    
    # Create small buffer to demonstrate overflow
    buffer = AsyncBuffer(maxsize=10, name='demo_buffer')
    
    # Push many items (some will be dropped)
    print("\n1. Pushing 20 items to buffer (size=10)...")
    for i in range(20):
        success = buffer.push({'item': i, 'value': i * 2})
        if not success and i < 15:  # Only print first few drops
            print(f"   Dropped item {i}")
    
    print(f"   Buffer size: {len(buffer)}/{buffer.maxsize}")
    print(f"   Stats: {buffer.stats()}")
    
    # Pop items
    print("\n2. Popping items...")
    for i in range(5):
        item = buffer.pop(timeout=0.1)
        if item:
            print(f"   Popped: {item}")
    
    print(f"\n3. Final stats:")
    stats = buffer.stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("✓ Buffer demo complete")


# ============================================================================
# Demo 2: AsyncMetricsWriter - Background Parquet Writer
# ============================================================================

async def demo_metrics_writer():
    """Demo background Parquet writer."""
    print("\n" + "="*70)
    print("DEMO 2: AsyncMetricsWriter - Background Parquet Writer")
    print("="*70)
    
    output_dir = Path('demo_output')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'metrics.parquet'
    
    # Create buffer and writer
    buffer = AsyncBuffer(maxsize=1000, name='metrics')
    writer = AsyncMetricsWriter(
        buffer=buffer,
        output_path=output_path,
        flush_interval=10,  # Flush every 10 items
        flush_seconds=2.0,
    )
    
    print(f"\n1. Starting writer (output: {output_path})...")
    writer.start()
    
    # Push metrics
    print("2. Pushing 50 metrics...")
    for i in range(50):
        buffer.push({
            'step': i,
            'loss': 1.0 / (i + 1),
            'accuracy': i / 50,
            'timestamp': time.time(),
        })
        await asyncio.sleep(0.01)  # Simulate training
    
    # Wait for flushes
    print("3. Waiting for background writes...")
    await asyncio.sleep(3.0)
    
    print(f"\n4. Writer stats:")
    stats = writer.stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Stop writer
    print("\n5. Stopping writer...")
    writer.stop()
    
    # Read back data
    print("\n6. Reading back data...")
    import pandas as pd
    df = pd.read_parquet(output_path)
    print(f"   Rows read: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    print(f"\n   First 5 rows:")
    print(df.head())
    
    print("✓ Metrics writer demo complete")


# ============================================================================
# Demo 3: CheckpointManager - Zarr Checkpointing
# ============================================================================

async def demo_checkpoint_manager():
    """Demo Zarr-based checkpointing."""
    print("\n" + "="*70)
    print("DEMO 3: CheckpointManager - Zarr Checkpointing")
    print("="*70)
    
    checkpoint_dir = Path('demo_output/checkpoints')
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    
    manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        keep_last=3,  # Keep only 3 checkpoints
        compression='zstd',
        compression_level=3,
    )
    
    print(f"\n1. Checkpoint dir: {checkpoint_dir}")
    
    # Save checkpoints
    print("\n2. Saving 5 checkpoints...")
    for epoch in range(1, 6):
        model_state = {
            'layer1_weights': np.random.randn(10, 10),
            'layer1_bias': np.random.randn(10),
            'layer2_weights': np.random.randn(10, 5),
            'layer2_bias': np.random.randn(5),
        }
        
        optimizer_state = {
            'learning_rate': 0.001,
            'momentum': np.random.randn(10, 10),
        }
        
        metadata = {
            'train_loss': 1.0 / epoch,
            'val_loss': 1.2 / epoch,
            'accuracy': epoch * 0.15,
        }
        
        await manager.save(
            epoch=epoch,
            model_state=model_state,
            optimizer_state=optimizer_state,
            metadata=metadata,
        )
    
    # List checkpoints
    print("\n3. Available checkpoints:")
    checkpoints = manager.list_checkpoints()
    print(f"   {checkpoints}")
    print(f"   (Kept only last 3 due to keep_last=3)")
    
    # Load checkpoint
    print("\n4. Loading checkpoint epoch 5...")
    state = await manager.load(epoch=5)
    print(f"   Epoch: {state['epoch']}")
    print(f"   Model keys: {list(state['model_state'].keys())}")
    print(f"   Optimizer keys: {list(state['optimizer_state'].keys())}")
    print(f"   Metadata: {state['metadata']}")
    
    # Get info
    print("\n5. Checkpoint info:")
    info = manager.get_checkpoint_info(epoch=5)
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    print("✓ Checkpoint manager demo complete")


# ============================================================================
# Demo 4: AsyncMetricsLogger - High-Level API
# ============================================================================

async def demo_metrics_logger():
    """Demo high-level metrics logger."""
    print("\n" + "="*70)
    print("DEMO 4: AsyncMetricsLogger - High-Level Logging API")
    print("="*70)
    
    output_path = Path('demo_output/training_metrics.parquet')
    
    logger = AsyncMetricsLogger(
        output_path=output_path,
        buffer_size=100,
        flush_interval=20,
    )
    
    print(f"\n1. Starting logger (output: {output_path})...")
    
    # Simulate training
    print("\n2. Simulating training (3 epochs, 10 steps each)...")
    for epoch in range(1, 4):
        print(f"\n   Epoch {epoch}:")
        
        for step in range(1, 11):
            # Log step
            await logger.log_step(
                epoch=epoch,
                step=step,
                train_loss=1.0 / (epoch * step),
                grad_norm=0.1 / step,
                circuits_executed=32,
                qubits_used=8,
                circuit_execution_time_ms=50.0,
                backend='ionq.simulator',
                throughput_samples_per_sec=100.0,
            )
            
            await asyncio.sleep(0.05)  # Simulate training time
        
        # Log epoch summary
        await logger.log_epoch(
            epoch=epoch,
            train_metrics={
                'loss': 0.5 / epoch,
                'grad_norm': 0.1,
                'circuit_time_ms': 500.0,
                'circuits_executed': 320,
                'qubits_used': 8,
                'backend': 'ionq.simulator',
            },
            val_metrics={
                'loss': 0.6 / epoch,
                'accuracy': epoch * 0.3,
            },
        )
        
        print(f"      Step logs: 10 steps")
        print(f"      Epoch summary logged")
    
    # Wait for writes
    print("\n3. Waiting for background writes...")
    await asyncio.sleep(2.0)
    
    print("\n4. Logger stats:")
    stats = logger.stats()
    print(f"   Buffer stats:")
    for key, value in stats['buffer'].items():
        print(f"     {key}: {value}")
    print(f"   Writer stats:")
    for key, value in stats['writer'].items():
        print(f"     {key}: {value}")
    
    # Stop logger
    print("\n5. Stopping logger...")
    logger.stop()
    
    # Read back
    print("\n6. Reading back training logs...")
    import pandas as pd
    df = pd.read_parquet(output_path)
    print(f"   Total rows: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    print(f"\n   Sample metrics:")
    print(df[['epoch', 'step', 'train_loss', 'grad_norm', 'backend']].head(10))
    
    # Query
    print(f"\n7. Querying epoch summaries (step=-1):")
    epoch_summaries = df[df['step'] == -1]
    print(epoch_summaries[['epoch', 'train_loss', 'val_loss']])
    
    print("✓ Metrics logger demo complete")


# ============================================================================
# Demo 5: Realistic Training Loop
# ============================================================================

async def demo_realistic_training():
    """Demo realistic training loop with all storage components."""
    print("\n" + "="*70)
    print("DEMO 5: Realistic Training Loop - All Components")
    print("="*70)
    
    # Setup
    output_dir = Path('demo_output/full_training')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = AsyncMetricsLogger(
        output_path=output_dir / 'metrics.parquet',
        buffer_size=500,
        flush_interval=50,
    )
    
    checkpoint_mgr = CheckpointManager(
        checkpoint_dir=output_dir / 'checkpoints',
        keep_last=2,
    )
    
    print(f"\n1. Training setup:")
    print(f"   Metrics: {output_dir / 'metrics.parquet'}")
    print(f"   Checkpoints: {output_dir / 'checkpoints'}")
    
    # Training loop
    print("\n2. Training 5 epochs...")
    for epoch in range(1, 6):
        epoch_start = time.time()
        
        # Simulate epoch training
        for step in range(1, 21):
            step_start = time.time()
            
            # Simulate quantum circuit execution
            await asyncio.sleep(0.02)  # Circuit time
            
            step_time = (time.time() - step_start) * 1000
            
            # Log metrics (non-blocking!)
            await logger.log_step(
                epoch=epoch,
                step=step,
                train_loss=1.0 / (epoch * step),
                grad_norm=np.random.rand() * 0.1,
                circuits_executed=32,
                qubits_used=8,
                circuit_execution_time_ms=20.0,
                batch_time_ms=step_time,
                throughput_samples_per_sec=32000.0 / step_time,
                backend='ionq.simulator',
            )
        
        epoch_time = time.time() - epoch_start
        
        # Epoch summary
        await logger.log_epoch(
            epoch=epoch,
            train_metrics={
                'loss': 0.5 / epoch,
                'grad_norm': 0.08,
                'circuit_time_ms': 400.0,
                'circuits_executed': 640,
            },
            val_metrics={
                'loss': 0.55 / epoch,
                'accuracy': 0.5 + epoch * 0.1,
            },
        )
        
        # Save checkpoint every 2 epochs
        if epoch % 2 == 0:
            model_state = {
                'layer1': np.random.randn(100, 100),
                'layer2': np.random.randn(100, 50),
            }
            await checkpoint_mgr.save(
                epoch=epoch,
                model_state=model_state,
                metadata={'loss': 0.5 / epoch, 'accuracy': 0.5 + epoch * 0.1},
            )
        
        print(f"   Epoch {epoch}: {epoch_time:.2f}s, loss={0.5/epoch:.4f}")
    
    # Cleanup
    print("\n3. Training complete, waiting for writes...")
    await asyncio.sleep(2.0)
    logger.stop()
    
    print("\n4. Final stats:")
    print(f"   Checkpoints saved: {checkpoint_mgr.list_checkpoints()}")
    print(f"   Metrics logged: {logger.stats()['writer']['rows_written']} rows")
    
    print("✓ Realistic training demo complete")


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run all demos."""
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║                  Q-Store v4.1 Phase 3 Storage Demo                ║")
    print("║                                                                    ║")
    print("║  Production-grade async storage with zero blocking!               ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    
    start_time = time.time()
    
    # Run demos
    await demo_async_buffer()
    await demo_metrics_writer()
    await demo_checkpoint_manager()
    await demo_metrics_logger()
    await demo_realistic_training()
    
    # Summary
    total_time = time.time() - start_time
    
    print("\n")
    print("="*70)
    print("PHASE 3 SUMMARY")
    print("="*70)
    print(f"✓ AsyncBuffer: Non-blocking ring buffer with overflow handling")
    print(f"✓ AsyncMetricsWriter: Background Parquet writer (Gzip compressed)")
    print(f"✓ CheckpointManager: Zarr checkpointing (Zstd compressed)")
    print(f"✓ AsyncMetricsLogger: High-level metrics API")
    print(f"✓ Full training loop: All components integrated")
    print(f"\nTotal demo time: {total_time:.2f}s")
    print(f"Output directory: demo_output/")
    print("\n" + "="*70)
    
    print("\n💡 Key Benefits:")
    print("   • Zero blocking - training never waits for I/O")
    print("   • Efficient storage - Zarr (binary) + Parquet (columnar)")
    print("   • Production-ready - buffer overflow, atomic saves, compression")
    print("   • Analytics-friendly - Parquet for pandas/SQL queries")
    
    print("\n📊 Next Steps:")
    print("   • Integrate with TensorFlow/PyTorch (Phase 4)")
    print("   • Add adaptive batch scheduling (Phase 5)")
    print("   • Create Fashion MNIST examples (Tasks 18-19)")


if __name__ == '__main__':
    asyncio.run(main())
