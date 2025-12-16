"""
Generic utility classes for Q-Store examples
Provides logging and benchmarking functionality
OPTIMIZED VERSION with reduced performance impact
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, TextIO
from contextlib import contextmanager
import time


class TeeStream:
    """
    Lightweight stream that writes to multiple outputs (like Unix tee command)
    Optimized for minimal performance impact with buffering
    """
    def __init__(self, *streams: TextIO):
        self.streams = streams
        self._buffer = []
        self._buffer_size = 0
        self._max_buffer = 8192  # Flush every 8KB

    def write(self, data: str):
        # Write to primary stream immediately (console)
        if self.streams:
            try:
                self.streams[0].write(data)
            except Exception:
                pass

        # Buffer writes to file to reduce I/O
        if len(self.streams) > 1:
            self._buffer.append(data)
            self._buffer_size += len(data)

            # Flush if buffer is large or ends with newline
            if self._buffer_size >= self._max_buffer or data.endswith('\n'):
                self._flush_buffer()

    def flush(self):
        if self.streams:
            try:
                self.streams[0].flush()
            except Exception:
                pass
        self._flush_buffer()

    def _flush_buffer(self):
        if not self._buffer or len(self.streams) <= 1:
            return

        try:
            combined = ''.join(self._buffer)
            for stream in self.streams[1:]:
                stream.write(combined)
                stream.flush()
        except Exception:
            pass
        finally:
            self._buffer.clear()
            self._buffer_size = 0


class ExampleLogger:
    """
    Comprehensive logging and benchmarking utility for Q-Store examples

    Features:
    - Creates LOG directory with timestamped log files
    - JSON benchmark tracking with start/finish times for all steps
    - Context manager for timing operations
    - Thread-safe logging
    - Optional stdout/stderr capture with buffering

    Performance optimizations:
    - Buffered I/O (8KB buffer)
    - Batch benchmark saves (every 5 steps instead of every step)
    - Atomic writes to prevent corruption
    - Reduced log levels (INFO instead of DEBUG)
    - Optional stdout capture (disabled by default)
    """

    def __init__(self,
                 log_dir: str = "LOG",
                 base_dir: Optional[str] = None,
                 example_name: str = "example",
                 capture_stdout: bool = False):
        """
        Initialize the example logger

        Args:
            log_dir: Directory name for logs (default: "LOG")
            base_dir: Base directory for logs (default: current working directory)
            example_name: Name of the example for file naming
            capture_stdout: Whether to capture print statements to log file (default: False for performance)
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.log_dir = self.base_dir / log_dir
        self.example_name = example_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.capture_stdout = capture_stdout

        # Create LOG directory
        self.log_dir.mkdir(exist_ok=True)

        # Setup log file
        self.log_file = self.log_dir / f"log-{self.timestamp}.txt"

        # Setup benchmark file
        self.benchmark_file = self.log_dir / f"benchmark-{self.example_name}-{self.timestamp}.json"

        # Benchmark data structure
        self.benchmarks: Dict[str, Any] = {
            "example_name": example_name,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "total_duration_ms": None,
            "steps": []
        }

        # Store original stdout/stderr
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._log_file_handle = None

        # Setup logger
        self._setup_logger()

        # Start capturing stdout/stderr if enabled
        if self.capture_stdout:
            self._start_capture()

        self.log_info(f"ExampleLogger initialized for '{example_name}'")
        self.log_info(f"Log directory: {self.log_dir}")
        self.log_info(f"Log file: {self.log_file}")
        self.log_info(f"Benchmark file: {self.benchmark_file}")

    def _setup_logger(self):
        """Configure the Python logger with optimized settings"""
        # Create a unique logger for this instance
        self.logger = logging.getLogger(f"q_store_examples.{self.example_name}.{self.timestamp}")
        self.logger.setLevel(logging.INFO)  # Changed to INFO to reduce overhead

        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()

        # File handler with buffering for better performance
        file_handler = logging.FileHandler(self.log_file, mode='w', buffering=8192)
        file_handler.setLevel(logging.INFO)

        # Console handler - only for warnings and errors to reduce noise
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)

        # Simpler formatter to reduce processing overhead
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _start_capture(self):
        """Start capturing stdout and stderr to log file with minimal overhead"""
        try:
            # Open log file in append mode with buffering
            self._log_file_handle = open(self.log_file, 'a', buffering=8192)

            # Create tee stream that writes to both console and file
            sys.stdout = TeeStream(self._original_stdout, self._log_file_handle)
            sys.stderr = TeeStream(self._original_stderr, self._log_file_handle)
        except Exception as e:
            self.logger.warning(f"Failed to capture stdout/stderr: {e}")

    def _stop_capture(self):
        """Stop capturing stdout and stderr"""
        try:
            # Flush any buffered output
            if hasattr(sys.stdout, '_flush_buffer'):
                sys.stdout._flush_buffer()
            if hasattr(sys.stderr, '_flush_buffer'):
                sys.stderr._flush_buffer()

            # Restore original streams
            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr

            # Close file handle
            if self._log_file_handle:
                self._log_file_handle.flush()
                self._log_file_handle.close()
                self._log_file_handle = None
        except Exception as e:
            self.logger.warning(f"Failed to stop stdout/stderr capture: {e}")

    def log_info(self, message: str):
        """Log an info message"""
        self.logger.info(message)

    def log_debug(self, message: str):
        """Log a debug message"""
        self.logger.debug(message)

    def log_warning(self, message: str):
        """Log a warning message"""
        self.logger.warning(message)

    def log_error(self, message: str, exc_info: bool = False):
        """Log an error message"""
        self.logger.error(message, exc_info=exc_info)

    def start_step(self, step_name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Start timing a benchmark step (lightweight operation)

        Args:
            step_name: Name of the step
            metadata: Optional metadata to store with the step
        """
        step_data = {
            "name": step_name,
            "start_time": datetime.now().isoformat(),
            "start_time_ms": time.time() * 1000,
            "end_time": None,
            "end_time_ms": None,
            "duration_ms": None,
            "status": "running",
            "metadata": metadata or {}
        }
        self.benchmarks["steps"].append(step_data)
        # Reduced logging overhead - only log to file
        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info(f"Started step: {step_name}")

        return len(self.benchmarks["steps"]) - 1  # Return step index

    def end_step(self, step_index: Optional[int] = None,
                 status: str = "completed",
                 result: Optional[Dict[str, Any]] = None):
        """
        End timing a benchmark step

        Args:
            step_index: Index of the step (default: last step)
            status: Status of the step completion ("completed", "failed", "skipped")
            result: Optional result data to store
        """
        if step_index is None:
            step_index = len(self.benchmarks["steps"]) - 1

        if step_index < 0 or step_index >= len(self.benchmarks["steps"]):
            self.log_warning(f"Invalid step index: {step_index}")
            return

        step = self.benchmarks["steps"][step_index]
        end_time_ms = time.time() * 1000

        step["end_time"] = datetime.now().isoformat()
        step["end_time_ms"] = end_time_ms
        step["duration_ms"] = end_time_ms - step["start_time_ms"]
        step["status"] = status

        if result:
            step["result"] = result

        # Reduced logging overhead - only log to file
        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info(f"Ended step: {step['name']} - Status: {status} - Duration: {step['duration_ms']:.2f}ms")

        # Save benchmark less frequently - every 5 steps or on failure for better performance
        if status == "failed" or len(self.benchmarks["steps"]) % 5 == 0:
            self._save_benchmark()

    @contextmanager
    def benchmark_step(self, step_name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for benchmarking a step

        Usage:
            with logger.benchmark_step("training"):
                # Your code here
                pass
        """
        step_index = self.start_step(step_name, metadata)
        try:
            yield
            self.end_step(step_index, status="completed")
        except Exception as e:
            self.log_error(f"Step '{step_name}' failed: {e}", exc_info=True)
            self.end_step(step_index, status="failed", result={"error": str(e)})
            raise

    def finalize(self):
        """Finalize the benchmark and save final data"""
        self.benchmarks["end_time"] = datetime.now().isoformat()

        # Calculate total duration
        if self.benchmarks["steps"]:
            start_ms = self.benchmarks["steps"][0]["start_time_ms"]
            end_ms = self.benchmarks["steps"][-1]["end_time_ms"]
            self.benchmarks["total_duration_ms"] = end_ms - start_ms if end_ms else None

        # Save final benchmark
        self._save_benchmark()

        self.log_info(f"Example '{self.example_name}' finalized")
        self.log_info(f"Total duration: {self.benchmarks['total_duration_ms']:.2f}ms")
        self.log_info(f"Benchmark saved to: {self.benchmark_file}")

        # Stop capturing stdout/stderr
        if self.capture_stdout:
            self._stop_capture()

    def __del__(self):
        """Cleanup when object is destroyed"""
        # Ensure streams are restored
        if hasattr(self, 'capture_stdout') and self.capture_stdout:
            try:
                if hasattr(self, '_original_stdout') and sys.stdout != self._original_stdout:
                    sys.stdout = self._original_stdout
                if hasattr(self, '_original_stderr') and sys.stderr != self._original_stderr:
                    sys.stderr = self._original_stderr
                if hasattr(self, '_log_file_handle') and self._log_file_handle:
                    self._log_file_handle.close()
            except Exception:
                pass  # Ignore errors during cleanup

    def _save_benchmark(self):
        """Save benchmark data to JSON file (optimized with atomic write)"""
        try:
            # Write to temporary file first for atomic operation
            temp_file = self.benchmark_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(self.benchmarks, f, indent=2)
            # Atomic rename
            temp_file.replace(self.benchmark_file)
        except Exception as e:
            if self.logger.isEnabledFor(logging.ERROR):
                self.logger.error(f"Failed to save benchmark: {e}")

    def get_step_summary(self) -> List[Dict[str, Any]]:
        """Get a summary of all benchmark steps"""
        summary = []
        for step in self.benchmarks["steps"]:
            summary.append({
                "name": step["name"],
                "duration_ms": step.get("duration_ms"),
                "status": step.get("status", "unknown")
            })
        return summary

    def print_summary(self):
        """Print a formatted summary of benchmarks"""
        print("\n" + "="*70)
        print(f"Benchmark Summary: {self.example_name}")
        print("="*70)

        for i, step in enumerate(self.benchmarks["steps"], 1):
            duration = step.get("duration_ms", 0)
            status = step.get("status", "unknown")
            print(f"{i}. {step['name']}")
            print(f"   Status: {status}")
            if duration:
                print(f"   Duration: {duration:.2f}ms")

        if self.benchmarks.get("total_duration_ms"):
            print(f"\nTotal Duration: {self.benchmarks['total_duration_ms']:.2f}ms")

        print("="*70 + "\n")


class BenchmarkCollector:
    """
    Collects benchmarks from multiple examples
    Useful for comparing performance across different runs
    """

    def __init__(self, log_dir: str = "LOG"):
        """Initialize the benchmark collector"""
        self.log_dir = Path(log_dir)
        self.benchmarks: List[Dict[str, Any]] = []

    def load_all_benchmarks(self):
        """Load all benchmark JSON files from the LOG directory"""
        if not self.log_dir.exists():
            return

        for file in self.log_dir.glob("benchmark-*.json"):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    self.benchmarks.append(data)
            except Exception as e:
                print(f"Error loading {file}: {e}")

    def compare_examples(self, example_name: Optional[str] = None):
        """
        Compare benchmarks, optionally filtered by example name

        Args:
            example_name: Filter by specific example name
        """
        filtered = self.benchmarks
        if example_name:
            filtered = [b for b in self.benchmarks if b.get("example_name") == example_name]

        if not filtered:
            print("No benchmarks found")
            return

        print("\n" + "="*70)
        print("Benchmark Comparison")
        print("="*70)

        for benchmark in filtered:
            print(f"\nExample: {benchmark.get('example_name')}")
            print(f"Date: {benchmark.get('start_time')}")
            print(f"Total Duration: {benchmark.get('total_duration_ms', 0):.2f}ms")
            print(f"Steps: {len(benchmark.get('steps', []))}")

        print("="*70 + "\n")
