"""
Data pipeline tools for hybrid quantum-classical workflows.

Provides efficient data processing pipelines for quantum computing.
"""

from typing import Callable, Optional, List, Dict, Any
import numpy as np
from ..core import UnifiedCircuit


class QuantumDataEncoder:
    """
    Encodes classical data for quantum processing.

    Handles data normalization, feature scaling, and quantum encoding.
    """

    def __init__(
        self,
        encoding_function: Callable[[np.ndarray], UnifiedCircuit],
        normalization: str = 'minmax'
    ):
        """
        Initialize data encoder.

        Args:
            encoding_function: Function to encode data into quantum circuit
            normalization: Normalization method ('minmax', 'standard', 'none')
        """
        self.encoding_function = encoding_function
        self.normalization = normalization
        self._min = None
        self._max = None
        self._mean = None
        self._std = None

    def fit(self, data: np.ndarray):
        """
        Fit normalization parameters.

        Args:
            data: Training data for fitting normalization
        """
        if self.normalization == 'minmax':
            self._min = np.min(data, axis=0)
            self._max = np.max(data, axis=0)
        elif self.normalization == 'standard':
            self._mean = np.mean(data, axis=0)
            self._std = np.std(data, axis=0)

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data.

        Args:
            data: Data to normalize

        Returns:
            Normalized data
        """
        if self.normalization == 'minmax':
            if self._min is None or self._max is None:
                return data
            range_vals = self._max - self._min
            range_vals[range_vals == 0] = 1  # Avoid division by zero
            return (data - self._min) / range_vals
        elif self.normalization == 'standard':
            if self._mean is None or self._std is None:
                return data
            std_vals = self._std.copy()
            std_vals[std_vals == 0] = 1  # Avoid division by zero
            return (data - self._mean) / std_vals
        return data

    def encode(self, data: np.ndarray) -> UnifiedCircuit:
        """
        Encode normalized data into quantum circuit.

        Args:
            data: Data to encode

        Returns:
            Quantum circuit with encoded data
        """
        normalized_data = self.normalize(data)
        return self.encoding_function(normalized_data)

    def __call__(self, data: np.ndarray) -> UnifiedCircuit:
        """Allow encoder to be called directly."""
        return self.encode(data)


class BatchProcessor:
    """
    Process data in batches for quantum circuits.

    Handles batching and parallel processing of quantum circuits.
    """

    def __init__(
        self,
        quantum_function: Callable,
        batch_size: int = 32
    ):
        """
        Initialize batch processor.

        Args:
            quantum_function: Function to process single data point
            batch_size: Size of batches
        """
        self.quantum_function = quantum_function
        self.batch_size = batch_size

    def process_batch(self, data: np.ndarray) -> List[Any]:
        """
        Process a batch of data.

        Args:
            data: Batch of data points

        Returns:
            List of quantum processing results
        """
        results = []
        for datapoint in data:
            result = self.quantum_function(datapoint)
            results.append(result)
        return results

    def process_dataset(self, dataset: np.ndarray) -> List[Any]:
        """
        Process entire dataset in batches.

        Args:
            dataset: Full dataset

        Returns:
            List of all results
        """
        n_samples = len(dataset)
        all_results = []

        for i in range(0, n_samples, self.batch_size):
            batch = dataset[i:i + self.batch_size]
            batch_results = self.process_batch(batch)
            all_results.extend(batch_results)

        return all_results

    def __call__(self, data: np.ndarray) -> List[Any]:
        """Allow processor to be called directly."""
        return self.process_dataset(data)


class ResultAggregator:
    """
    Aggregate results from quantum computations.

    Combines multiple quantum measurement results.
    """

    def __init__(self, aggregation_method: str = 'mean'):
        """
        Initialize result aggregator.

        Args:
            aggregation_method: Method for aggregation ('mean', 'sum', 'max', 'vote')
        """
        self.aggregation_method = aggregation_method

    def aggregate(self, results: List[np.ndarray]) -> np.ndarray:
        """
        Aggregate multiple results.

        Args:
            results: List of result arrays

        Returns:
            Aggregated result
        """
        if not results:
            return np.array([])

        results_array = np.array(results)

        if self.aggregation_method == 'mean':
            return np.mean(results_array, axis=0)
        elif self.aggregation_method == 'sum':
            return np.sum(results_array, axis=0)
        elif self.aggregation_method == 'max':
            return np.max(results_array, axis=0)
        elif self.aggregation_method == 'vote':
            # Majority voting for classification
            from scipy.stats import mode
            return mode(results_array, axis=0, keepdims=False)[0]
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

    def __call__(self, results: List[np.ndarray]) -> np.ndarray:
        """Allow aggregator to be called directly."""
        return self.aggregate(results)


class DataPipeline:
    """
    Complete data processing pipeline for hybrid workflows.

    Chains encoding, processing, and aggregation steps.
    """

    def __init__(
        self,
        encoder: Optional[QuantumDataEncoder] = None,
        processor: Optional[BatchProcessor] = None,
        aggregator: Optional[ResultAggregator] = None
    ):
        """
        Initialize data pipeline.

        Args:
            encoder: Data encoder
            processor: Batch processor
            aggregator: Result aggregator
        """
        self.encoder = encoder
        self.processor = processor
        self.aggregator = aggregator

    def run(self, data: np.ndarray) -> Any:
        """
        Run full pipeline on data.

        Args:
            data: Input data

        Returns:
            Processed and aggregated results
        """
        # Encoding stage
        if self.encoder is not None:
            encoded_data = []
            for datapoint in data:
                encoded_data.append(self.encoder(datapoint))
            data = encoded_data

        # Processing stage
        if self.processor is not None:
            results = self.processor(data)
        else:
            results = data

        # Aggregation stage
        if self.aggregator is not None:
            results = self.aggregator(results)

        return results

    def __call__(self, data: np.ndarray) -> Any:
        """Allow pipeline to be called directly."""
        return self.run(data)


def create_data_pipeline(
    encoding_function: Optional[Callable] = None,
    processing_function: Optional[Callable] = None,
    aggregation_method: str = 'mean',
    batch_size: int = 32
) -> DataPipeline:
    """
    Create a data processing pipeline.

    Args:
        encoding_function: Function to encode data
        processing_function: Function to process encoded data
        aggregation_method: Method for result aggregation
        batch_size: Batch size for processing

    Returns:
        Configured data pipeline
    """
    encoder = None
    if encoding_function is not None:
        encoder = QuantumDataEncoder(encoding_function)

    processor = None
    if processing_function is not None:
        processor = BatchProcessor(processing_function, batch_size)

    aggregator = ResultAggregator(aggregation_method)

    return DataPipeline(encoder, processor, aggregator)
