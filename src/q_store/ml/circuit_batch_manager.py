"""
Circuit Batch Manager - v3.3
Batches multiple circuit executions for parallel processing

Key Innovation: Reduces API overhead by batching circuit submissions
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from queue import Queue
from collections import defaultdict

from ..backends.quantum_backend_interface import (
    QuantumBackend,
    QuantumCircuit,
    ExecutionResult
)

logger = logging.getLogger(__name__)


@dataclass
class CircuitJob:
    """Represents a submitted circuit job"""
    job_id: str
    circuit: QuantumCircuit
    shots: int
    status: str  # 'submitted', 'running', 'completed', 'failed'
    submit_time: float
    complete_time: Optional[float] = None
    result: Optional[ExecutionResult] = None
    error: Optional[str] = None


class CircuitBatchManager:
    """
    Batches multiple circuit executions into single API calls

    Problem: Submitting 96 circuits one-by-one has massive overhead
    Solution: Submit all 96 as a batch, poll for results

    Benefits:
    - Amortize API latency
    - Reduce queue wait time
    - Enable parallel execution on quantum hardware
    """

    def __init__(
        self,
        backend: QuantumBackend,
        max_batch_size: int = 100,
        polling_interval: float = 0.5,
        timeout: float = 300.0
    ):
        """
        Initialize batch manager

        Args:
            backend: Quantum backend for execution
            max_batch_size: Maximum circuits per batch
            polling_interval: Seconds between status checks
            timeout: Maximum wait time for batch completion
        """
        self.backend = backend
        self.max_batch_size = max_batch_size
        self.polling_interval = polling_interval
        self.timeout = timeout

        # Active job tracking
        self._active_jobs: Dict[str, CircuitJob] = {}
        self._job_queue: Queue = Queue()

        # Statistics
        self.total_circuits_submitted = 0
        self.total_circuits_completed = 0
        self.total_wait_time_ms = 0.0

    async def execute_batch(
        self,
        circuits: List[QuantumCircuit],
        shots: int = 1000,
        wait_for_results: bool = True
    ) -> List[ExecutionResult]:
        """
        Execute multiple circuits efficiently

        Strategy:
        1. Submit all circuits as separate jobs (non-blocking)
        2. Poll for results asynchronously
        3. Return results as they complete

        Args:
            circuits: List of circuits to execute
            shots: Shots per circuit
            wait_for_results: If True, wait for completion; else return job IDs

        Returns:
            List of ExecutionResults in same order as circuits
        """
        if not circuits:
            return []

        batch_start = time.time()

        # Split into batches if needed
        batches = self._split_into_batches(circuits)

        job_ids = []

        # Submit all batches
        for batch in batches:
            batch_job_ids = await self._submit_batch(batch, shots)
            job_ids.extend(batch_job_ids)

        if not wait_for_results:
            return job_ids

        # Poll for results
        results = await self._poll_for_results(job_ids)

        batch_time = (time.time() - batch_start) * 1000
        self.total_wait_time_ms += batch_time

        logger.info(
            f"Batch execution completed: {len(circuits)} circuits in "
            f"{batch_time:.2f}ms ({batch_time/len(circuits):.2f}ms per circuit)"
        )

        return results

    def _split_into_batches(
        self,
        circuits: List[QuantumCircuit]
    ) -> List[List[QuantumCircuit]]:
        """Split circuits into batches of max_batch_size"""
        batches = []
        for i in range(0, len(circuits), self.max_batch_size):
            batches.append(circuits[i:i + self.max_batch_size])
        return batches

    async def _submit_batch(
        self,
        circuits: List[QuantumCircuit],
        shots: int
    ) -> List[str]:
        """Submit a batch of circuits"""
        job_ids = []

        # Submit all circuits in parallel
        submit_tasks = [
            self._submit_job_async(circuit, shots)
            for circuit in circuits
        ]

        job_ids = await asyncio.gather(*submit_tasks)

        self.total_circuits_submitted += len(circuits)

        logger.debug(f"Submitted batch of {len(circuits)} circuits")

        return job_ids

    async def _submit_job_async(
        self,
        circuit: QuantumCircuit,
        shots: int
    ) -> str:
        """Submit single job without waiting for completion"""
        submit_start = time.time()

        # Check if backend supports async submission
        if hasattr(self.backend, 'submit_job_async'):
            job_id = await self.backend.submit_job_async(circuit, shots)

            # Track job
            self._active_jobs[job_id] = CircuitJob(
                job_id=job_id,
                circuit=circuit,
                shots=shots,
                status='submitted',
                submit_time=submit_start
            )
        else:
            # Fallback: execute immediately for backends without job submission
            job_id = f"immediate_{id(circuit)}_{time.time()}"

            # Execute circuit directly
            result = await self.backend.execute_circuit(circuit, shots)

            # Store result directly
            self._active_jobs[job_id] = CircuitJob(
                job_id=job_id,
                circuit=circuit,
                shots=shots,
                status='completed',
                submit_time=submit_start,
                complete_time=time.time(),
                result=result
            )

        return job_id

    async def _poll_for_results(
        self,
        job_ids: List[str]
    ) -> List[ExecutionResult]:
        """Poll for job completion and collect results"""
        results = {}
        pending = set(job_ids)
        start_time = time.time()

        while pending:
            # Check timeout
            if time.time() - start_time > self.timeout:
                logger.error(f"Batch timeout: {len(pending)} jobs still pending")
                raise TimeoutError(
                    f"Batch execution timeout after {self.timeout}s"
                )

            # Check all pending jobs
            for job_id in list(pending):
                job = self._active_jobs.get(job_id)

                if not job:
                    logger.error(f"Job {job_id} not found in active jobs")
                    pending.remove(job_id)
                    continue

                # If already completed (sync execution)
                if job.status == 'completed' and job.result is not None:
                    results[job_id] = job.result
                    pending.remove(job_id)
                    self.total_circuits_completed += 1
                    continue

                # Check job status
                status = await self._check_job_status(job_id)

                if status == 'completed':
                    result = await self._fetch_result(job_id)
                    results[job_id] = result
                    pending.remove(job_id)
                    self.total_circuits_completed += 1

                    job.status = 'completed'
                    job.complete_time = time.time()
                    job.result = result

                elif status == 'failed':
                    logger.error(f"Job {job_id} failed")
                    pending.remove(job_id)

                    job.status = 'failed'
                    job.complete_time = time.time()

                    # Create a dummy error result
                    results[job_id] = ExecutionResult(
                        measurements={},
                        metadata={'error': 'Job failed'}
                    )

            # Wait before next poll
            if pending:
                await asyncio.sleep(self.polling_interval)

        # Return results in order
        return [results[job_id] for job_id in job_ids]

    async def _check_job_status(self, job_id: str) -> str:
        """Check status of a job"""
        job = self._active_jobs.get(job_id)

        if not job:
            return 'unknown'

        # If already completed
        if job.status == 'completed':
            return 'completed'

        # Check backend status
        if hasattr(self.backend, 'check_job_status'):
            try:
                status = await self.backend.check_job_status(job_id)
                job.status = status
                return status
            except Exception as e:
                logger.error(f"Error checking job status: {e}")
                return 'failed'

        # Fallback: assume completed
        return 'completed'

    async def _fetch_result(self, job_id: str) -> ExecutionResult:
        """Fetch result for completed job"""
        job = self._active_jobs.get(job_id)

        if not job:
            raise ValueError(f"Job {job_id} not found")

        # If result already cached
        if job.result is not None:
            return job.result

        # Fetch from backend
        if hasattr(self.backend, 'get_job_result'):
            try:
                result = await self.backend.get_job_result(job_id)
                return result
            except Exception as e:
                logger.error(f"Error fetching job result: {e}")
                raise

        # Fallback: execute synchronously
        logger.warning(f"Backend doesn't support async result fetching, executing synchronously")
        result = await self.backend.execute_circuit(job.circuit, job.shots)
        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get batch manager statistics"""
        avg_wait = (
            self.total_wait_time_ms / self.total_circuits_completed
            if self.total_circuits_completed > 0 else 0
        )

        return {
            'total_circuits_submitted': self.total_circuits_submitted,
            'total_circuits_completed': self.total_circuits_completed,
            'total_wait_time_ms': self.total_wait_time_ms,
            'avg_wait_time_ms': avg_wait,
            'active_jobs': len(self._active_jobs),
            'success_rate': (
                self.total_circuits_completed / self.total_circuits_submitted
                if self.total_circuits_submitted > 0 else 0
            )
        }

    def clear_completed_jobs(self, older_than_seconds: float = 300):
        """Clear completed jobs older than specified time"""
        current_time = time.time()
        to_remove = []

        for job_id, job in self._active_jobs.items():
            if job.status in ['completed', 'failed']:
                if job.complete_time and (current_time - job.complete_time) > older_than_seconds:
                    to_remove.append(job_id)

        for job_id in to_remove:
            del self._active_jobs[job_id]

        if to_remove:
            logger.debug(f"Cleared {len(to_remove)} completed jobs")
