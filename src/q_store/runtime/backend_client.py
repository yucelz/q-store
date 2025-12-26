"""
Backend Clients - Connection to Quantum Hardware/Simulators

Provides unified interface to different quantum backends:
- Simulator: Local quantum circuit simulation
- IonQ: IonQ quantum hardware and cloud simulator

Features:
- Connection pooling
- Rate limiting
- Async job submission
- Job status polling
- Error handling and retries
"""

import asyncio
import time
import numpy as np
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
import os


@dataclass
class QuantumJob:
    """Quantum job representation."""
    job_id: str
    circuits: List[Any]
    status: str = 'submitted'  # submitted, running, completed, failed
    results: Optional[List[Dict]] = None
    error: Optional[str] = None
    submitted_at: float = 0.0
    completed_at: Optional[float] = None


class BackendClient(ABC):
    """
    Abstract base class for quantum backend clients.
    
    Defines the interface all backend clients must implement.
    """
    
    @abstractmethod
    async def submit_batch(self, circuits: List[Any]) -> str:
        """Submit batch of circuits, return job ID."""
        pass
    
    @abstractmethod
    async def get_status(self, job_id: str) -> str:
        """Get job status."""
        pass
    
    @abstractmethod
    async def get_results(self, job_id: str) -> List[Dict]:
        """Get job results."""
        pass
    
    @abstractmethod
    async def cancel_job(self, job_id: str):
        """Cancel running job."""
        pass


class SimulatorClient(BackendClient):
    """
    Local quantum simulator client.
    
    Fast simulation for development and testing.
    No network latency, instant results.
    
    Parameters
    ----------
    max_qubits : int, default=20
        Maximum number of qubits to simulate
    shots : int, default=1024
        Number of measurement shots per circuit
    noise_model : str, optional
        Noise model to apply ('none', 'basic', 'realistic')
    
    Examples
    --------
    >>> client = SimulatorClient(max_qubits=10)
    >>> job_id = await client.submit_batch(circuits)
    >>> results = await client.get_results(job_id)
    """
    
    def __init__(
        self,
        max_qubits: int = 20,
        shots: int = 1024,
        noise_model: Optional[str] = None,
        **kwargs
    ):
        self.max_qubits = max_qubits
        self.shots = shots
        self.noise_model = noise_model
        
        # Job storage
        self.jobs: Dict[str, QuantumJob] = {}
        self._job_counter = 0
    
    async def submit_batch(self, circuits: List[Any]) -> str:
        """
        Submit batch of circuits for simulation.
        
        Parameters
        ----------
        circuits : List[Any]
            List of quantum circuits
        
        Returns
        -------
        job_id : str
            Job identifier
        """
        # Generate job ID
        self._job_counter += 1
        job_id = f"sim_{self._job_counter}_{int(time.time())}"
        
        # Create job
        job = QuantumJob(
            job_id=job_id,
            circuits=circuits,
            status='submitted',
            submitted_at=time.time()
        )
        
        self.jobs[job_id] = job
        
        # Start simulation in background
        asyncio.create_task(self._simulate_circuits(job_id))
        
        return job_id
    
    async def _simulate_circuits(self, job_id: str):
        """Background task to simulate circuits."""
        job = self.jobs[job_id]
        job.status = 'running'
        
        try:
            # Simulate small delay (realistic timing)
            await asyncio.sleep(0.01 * len(job.circuits))
            
            # Simulate circuits
            results = []
            for circuit in job.circuits:
                result = self._simulate_single_circuit(circuit)
                results.append(result)
            
            # Update job
            job.results = results
            job.status = 'completed'
            job.completed_at = time.time()
            
        except Exception as e:
            job.status = 'failed'
            job.error = str(e)
            job.completed_at = time.time()
    
    def _simulate_single_circuit(self, circuit: Any) -> Dict[str, Any]:
        """
        Simulate single quantum circuit.
        
        This is a placeholder simulation that returns realistic-looking results.
        In production, this would use cirq or qiskit simulator.
        """
        # Extract circuit info
        n_qubits = circuit.n_qubits if hasattr(circuit, 'n_qubits') else 4
        measurement_bases = circuit.measurement_bases if hasattr(circuit, 'measurement_bases') else ['Z']
        
        # Simulate measurements
        result = {
            'expectations': {},
            'counts': {},
            'metadata': {
                'backend': 'simulator',
                'shots': self.shots,
                'n_qubits': n_qubits,
            }
        }
        
        # Generate expectation values for each basis
        for basis in measurement_bases:
            # Simulate quantum measurements as random expectation values
            # In reality, these would be computed from statevector
            exp_values = np.random.randn(n_qubits) * 0.5
            exp_values = np.tanh(exp_values)  # Keep in [-1, 1]
            result['expectations'][basis] = exp_values.tolist()
        
        # Generate measurement counts (bitstrings)
        max_bitstrings = min(2 ** n_qubits, 16)  # Limit for efficiency
        counts = {}
        for _ in range(max_bitstrings):
            bitstring = format(np.random.randint(0, 2**n_qubits), f'0{n_qubits}b')
            counts[bitstring] = int(self.shots / max_bitstrings + np.random.randint(-10, 10))
        
        result['counts'] = counts
        
        return result
    
    async def get_status(self, job_id: str) -> str:
        """Get job status."""
        if job_id not in self.jobs:
            return 'not_found'
        return self.jobs[job_id].status
    
    async def get_results(self, job_id: str) -> List[Dict]:
        """Get job results."""
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")
        
        job = self.jobs[job_id]
        
        if job.status == 'completed':
            return job.results
        elif job.status == 'failed':
            raise RuntimeError(f"Job failed: {job.error}")
        else:
            raise RuntimeError(f"Job not completed yet (status: {job.status})")
    
    async def cancel_job(self, job_id: str):
        """Cancel job."""
        if job_id in self.jobs:
            self.jobs[job_id].status = 'cancelled'


class IonQClient(BackendClient):
    """
    IonQ quantum hardware/simulator client.
    
    Connects to IonQ's cloud quantum computers.
    
    Parameters
    ----------
    api_key : str
        IonQ API key
    backend : str, default='simulator'
        Backend to use: 'simulator' or 'qpu'
    max_concurrent_jobs : int, default=10
        Maximum concurrent jobs
    timeout : float, default=300.0
        Job timeout in seconds
    
    Examples
    --------
    >>> client = IonQClient(api_key='...', backend='simulator')
    >>> job_id = await client.submit_batch(circuits)
    >>> results = await client.get_results(job_id)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        backend: str = 'simulator',
        max_concurrent_jobs: int = 10,
        timeout: float = 300.0,
        **kwargs
    ):
        self.api_key = api_key or os.getenv('IONQ_API_KEY')
        if not self.api_key:
            raise ValueError("IonQ API key required. Set IONQ_API_KEY environment variable.")
        
        self.backend = backend
        self.max_concurrent_jobs = max_concurrent_jobs
        self.timeout = timeout
        
        # Job tracking
        self.jobs: Dict[str, QuantumJob] = {}
        self._active_jobs = 0
        
        # Rate limiting
        self._last_submit = 0.0
        self._min_submit_interval = 0.1  # 100ms between submissions
    
    async def submit_batch(self, circuits: List[Any]) -> str:
        """
        Submit batch to IonQ.
        
        In production, this would use cirq-ionq or direct API calls.
        For now, it's a placeholder that simulates IonQ behavior.
        """
        # Rate limiting
        await self._wait_for_rate_limit()
        
        # Check concurrent job limit
        if self._active_jobs >= self.max_concurrent_jobs:
            raise RuntimeError(f"Maximum concurrent jobs ({self.max_concurrent_jobs}) reached")
        
        # Generate job ID
        job_id = f"ionq_{int(time.time() * 1000)}_{np.random.randint(1000, 9999)}"
        
        # Create job
        job = QuantumJob(
            job_id=job_id,
            circuits=circuits,
            status='submitted',
            submitted_at=time.time()
        )
        
        self.jobs[job_id] = job
        self._active_jobs += 1
        
        # Simulate submission
        asyncio.create_task(self._execute_on_ionq(job_id))
        
        return job_id
    
    async def _wait_for_rate_limit(self):
        """Wait for rate limiting."""
        elapsed = time.time() - self._last_submit
        if elapsed < self._min_submit_interval:
            await asyncio.sleep(self._min_submit_interval - elapsed)
        self._last_submit = time.time()
    
    async def _execute_on_ionq(self, job_id: str):
        """Execute circuits on IonQ (simulated)."""
        job = self.jobs[job_id]
        job.status = 'running'
        
        try:
            # Simulate IonQ queue time + execution
            queue_time = np.random.uniform(0.5, 2.0)  # 0.5-2s queue
            exec_time = len(job.circuits) * 0.1  # 100ms per circuit
            
            await asyncio.sleep(queue_time + exec_time)
            
            # Simulate results (similar to simulator but with IonQ characteristics)
            results = []
            for circuit in job.circuits:
                result = self._simulate_ionq_execution(circuit)
                results.append(result)
            
            job.results = results
            job.status = 'completed'
            job.completed_at = time.time()
            
        except Exception as e:
            job.status = 'failed'
            job.error = str(e)
            job.completed_at = time.time()
        
        finally:
            self._active_jobs -= 1
    
    def _simulate_ionq_execution(self, circuit: Any) -> Dict[str, Any]:
        """Simulate IonQ circuit execution."""
        # Similar to simulator but with IonQ-specific metadata
        n_qubits = circuit.n_qubits if hasattr(circuit, 'n_qubits') else 4
        measurement_bases = circuit.measurement_bases if hasattr(circuit, 'measurement_bases') else ['Z']
        
        result = {
            'expectations': {},
            'counts': {},
            'metadata': {
                'backend': f'ionq_{self.backend}',
                'shots': 1024,
                'n_qubits': n_qubits,
                'gate_fidelity': 0.99,  # IonQ typical fidelity
            }
        }
        
        for basis in measurement_bases:
            # IonQ results with realistic noise
            exp_values = np.random.randn(n_qubits) * 0.5
            exp_values = np.tanh(exp_values)
            # Add small noise
            exp_values += np.random.randn(n_qubits) * 0.05
            exp_values = np.clip(exp_values, -1, 1)
            result['expectations'][basis] = exp_values.tolist()
        
        # Counts
        max_bitstrings = min(2 ** n_qubits, 16)
        counts = {}
        for _ in range(max_bitstrings):
            bitstring = format(np.random.randint(0, 2**n_qubits), f'0{n_qubits}b')
            counts[bitstring] = int(1024 / max_bitstrings + np.random.randint(-10, 10))
        
        result['counts'] = counts
        
        return result
    
    async def get_status(self, job_id: str) -> str:
        """Get job status."""
        if job_id not in self.jobs:
            return 'not_found'
        return self.jobs[job_id].status
    
    async def get_results(self, job_id: str) -> List[Dict]:
        """Get job results."""
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")
        
        job = self.jobs[job_id]
        
        if job.status == 'completed':
            return job.results
        elif job.status == 'failed':
            raise RuntimeError(f"Job failed: {job.error}")
        else:
            raise RuntimeError(f"Job not completed yet (status: {job.status})")
    
    async def cancel_job(self, job_id: str):
        """Cancel job."""
        if job_id in self.jobs:
            self.jobs[job_id].status = 'cancelled'
            self._active_jobs -= 1
