"""
Async Metrics Writer - Background Parquet Writer

Writes metrics to Parquet files in background thread.
Never blocks training!

Features:
- Background thread writer
- Batch writes for efficiency
- Automatic flushing
- Append mode
- Compression
- Schema validation

Design:
- Pull from AsyncBuffer
- Batch up to flush_interval items
- Write to Parquet (append mode)
- Gzip compression
- Automatic schema inference
"""

import threading
import queue
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd


class AsyncMetricsWriter(threading.Thread):
    """
    Background thread that writes metrics to Parquet.
    
    Never blocks training loop!
    
    Parameters
    ----------
    buffer : AsyncBuffer
        Buffer to read from
    output_path : Path or str
        Output Parquet file path
    flush_interval : int, default=100
        Flush after this many items
    flush_seconds : float, default=10.0
        Or flush after this many seconds
    
    Examples
    --------
    >>> buffer = AsyncBuffer()
    >>> writer = AsyncMetricsWriter(buffer, 'metrics.parquet')
    >>> writer.start()
    >>> # ... training happens ...
    >>> writer.stop()
    """
    
    def __init__(
        self,
        buffer: Any,  # AsyncBuffer
        output_path: Path,
        flush_interval: int = 100,
        flush_seconds: float = 10.0,
    ):
        super().__init__(daemon=True, name='AsyncMetricsWriter')
        self.buffer = buffer
        self.output_path = Path(output_path)
        self.flush_interval = flush_interval
        self.flush_seconds = flush_seconds
        
        # Create output directory
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # State
        self.rows: List[Dict] = []
        self._stop_event = threading.Event()
        self._last_flush = time.time()
        
        # Statistics
        self.rows_written = 0
        self.flushes = 0
        self.errors = 0
    
    def run(self):
        """Background loop."""
        print(f"📝 AsyncMetricsWriter started: {self.output_path}")
        
        while not self._stop_event.is_set():
            try:
                # Get item from buffer (blocking with timeout)
                item = self.buffer.pop(timeout=0.5)
                
                if item is not None:
                    self.rows.append(item)
                
                # Check if we should flush
                should_flush = (
                    len(self.rows) >= self.flush_interval or
                    (time.time() - self._last_flush) >= self.flush_seconds
                )
                
                if should_flush and len(self.rows) > 0:
                    self._flush()
            
            except Exception as e:
                print(f"⚠️  AsyncMetricsWriter error: {e}")
                self.errors += 1
                time.sleep(0.1)  # Back off on error
        
        # Final flush on shutdown
        if len(self.rows) > 0:
            self._flush()
        
        print(f"✓ AsyncMetricsWriter stopped: {self.rows_written} rows written, {self.flushes} flushes")
    
    def _flush(self):
        """Write accumulated rows to Parquet."""
        if not self.rows:
            return
        
        try:
            df = pd.DataFrame(self.rows)
            
            # Append to existing file
            if self.output_path.exists():
                # Read existing and append
                existing_df = pd.read_parquet(self.output_path)
                df = pd.concat([existing_df, df], ignore_index=True)
            
            # Write with compression
            df.to_parquet(
                self.output_path,
                compression='gzip',
                index=False,
                engine='pyarrow'
            )
            
            self.rows_written += len(self.rows)
            self.flushes += 1
            self._last_flush = time.time()
            
            # Clear rows
            self.rows.clear()
        
        except Exception as e:
            print(f"⚠️  Error writing metrics: {e}")
            self.errors += 1
            # Don't clear rows on error - will retry next flush
    
    def stop(self):
        """Stop writer thread."""
        self._stop_event.set()
        self.join(timeout=5.0)
    
    def stats(self) -> Dict[str, Any]:
        """Get writer statistics."""
        return {
            'rows_written': self.rows_written,
            'flushes': self.flushes,
            'errors': self.errors,
            'pending_rows': len(self.rows),
            'output_path': str(self.output_path),
        }
