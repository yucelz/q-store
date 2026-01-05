"""
Q-Store Backend API Client.

This module provides a REST client for accessing the Q-Store Backend API,
enabling dataset management, HuggingFace imports, Label Studio integration,
and data augmentation features.

The BackendAPIClient handles:
- Authentication (API keys, JWT tokens)
- Dataset listing and retrieval
- Dataset data download
- HuggingFace dataset imports
- Label Studio integration
- Data augmentation via Albumentations
- Error handling and logging

Example:
    >>> from q_store.data.backend_client import BackendAPIClient
    >>>
    >>> # Initialize client
    >>> client = BackendAPIClient(
    ...     base_url="http://localhost:8000",
    ...     api_key="your_api_key"
    ... )
    >>>
    >>> # List available datasets
    >>> datasets = client.list_datasets()
    >>>
    >>> # Get dataset details
    >>> dataset = client.get_dataset(dataset_id="uuid-123")
    >>>
    >>> # Download dataset data
    >>> data = client.download_dataset_data(dataset_id="uuid-123", split="train")
"""

import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import io

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class BackendDatasetInfo:
    """Container for backend dataset information."""

    id: str
    name: str
    source: str  # 'huggingface', 'label_studio', 'custom', etc.
    description: Optional[str] = None
    num_samples: Optional[int] = None
    num_classes: Optional[int] = None
    splits: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __repr__(self) -> str:
        return (
            f"BackendDatasetInfo(id='{self.id}', name='{self.name}', "
            f"source='{self.source}', samples={self.num_samples}, "
            f"classes={self.num_classes})"
        )


class BackendAPIClient:
    """
    REST client for Q-Store Backend API.

    Provides methods for:
    - Dataset management (list, get, delete)
    - Dataset data download
    - HuggingFace dataset imports
    - Label Studio integration
    - Data augmentation

    Attributes:
        base_url (str): Backend API base URL (e.g., "http://localhost:8000")
        api_key (Optional[str]): API key for authentication
        jwt_token (Optional[str]): JWT token for authentication
        timeout (int): Request timeout in seconds (default: 30)
        verify_ssl (bool): Whether to verify SSL certificates (default: True)

    Example:
        >>> client = BackendAPIClient(
        ...     base_url="http://localhost:8000",
        ...     api_key="your_api_key"
        ... )
        >>> datasets = client.list_datasets()
        >>> for ds in datasets:
        ...     print(f"{ds.name}: {ds.num_samples} samples")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        jwt_token: Optional[str] = None,
        timeout: int = 30,
        verify_ssl: bool = True
    ):
        """
        Initialize Backend API client.

        Args:
            base_url: Backend API base URL
            api_key: Optional API key for authentication
            jwt_token: Optional JWT token for authentication
            timeout: Request timeout in seconds
            verify_ssl: Whether to verify SSL certificates

        Raises:
            ImportError: If requests library is not available
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError(
                "requests library is required for BackendAPIClient. "
                "Install it with: pip install requests"
            )

        if not NUMPY_AVAILABLE:
            raise ImportError(
                "numpy library is required for BackendAPIClient. "
                "Install it with: pip install numpy"
            )

        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.jwt_token = jwt_token
        self.timeout = timeout
        self.verify_ssl = verify_ssl

        # Session for connection pooling
        self.session = requests.Session()

        # Set up authentication headers
        if self.jwt_token:
            self.session.headers.update({
                'Authorization': f'Bearer {self.jwt_token}'
            })
        elif self.api_key:
            self.session.headers.update({
                'X-API-Key': self.api_key
            })

        logger.info(f"Initialized BackendAPIClient for {self.base_url}")

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> requests.Response:
        """
        Make HTTP request to backend API.

        Args:
            method: HTTP method ('GET', 'POST', 'PUT', 'DELETE')
            endpoint: API endpoint (e.g., '/datasets')
            params: Query parameters
            json: JSON body for POST/PUT requests
            **kwargs: Additional arguments passed to requests

        Returns:
            Response object

        Raises:
            requests.RequestException: If request fails
        """
        url = f"{self.base_url}{endpoint}"

        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=json,
                timeout=self.timeout,
                verify=self.verify_ssl,
                **kwargs
            )
            response.raise_for_status()
            return response

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error {response.status_code} for {method} {url}: {e}")
            raise
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error for {method} {url}: {e}")
            raise
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout error for {method} {url}: {e}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {method} {url}: {e}")
            raise

    def list_datasets(
        self,
        source: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[BackendDatasetInfo]:
        """
        List available datasets from backend.

        Args:
            source: Filter by source ('huggingface', 'label_studio', 'custom')
            limit: Maximum number of datasets to return
            offset: Number of datasets to skip (for pagination)

        Returns:
            List of BackendDatasetInfo objects

        Example:
            >>> datasets = client.list_datasets(source='huggingface', limit=10)
            >>> for ds in datasets:
            ...     print(f"{ds.name}: {ds.num_samples} samples")
        """
        params = {'offset': offset}
        if source:
            params['source'] = source
        if limit:
            params['limit'] = limit

        response = self._request('GET', '/datasets', params=params)
        datasets_data = response.json()

        datasets = []
        for ds_data in datasets_data:
            dataset = BackendDatasetInfo(
                id=ds_data['id'],
                name=ds_data['name'],
                source=ds_data.get('source', 'unknown'),
                description=ds_data.get('description'),
                num_samples=ds_data.get('num_samples'),
                num_classes=ds_data.get('num_classes'),
                splits=ds_data.get('splits', []),
                metadata=ds_data.get('metadata', {})
            )
            datasets.append(dataset)

        logger.info(f"Listed {len(datasets)} datasets from backend")
        return datasets

    def get_dataset(self, dataset_id: str) -> BackendDatasetInfo:
        """
        Get detailed information about a specific dataset.

        Args:
            dataset_id: UUID of the dataset

        Returns:
            BackendDatasetInfo object with dataset details

        Raises:
            requests.HTTPError: If dataset not found (404)

        Example:
            >>> dataset = client.get_dataset("uuid-123")
            >>> print(f"{dataset.name} has {dataset.num_samples} samples")
        """
        response = self._request('GET', f'/datasets/{dataset_id}')
        ds_data = response.json()

        dataset = BackendDatasetInfo(
            id=ds_data['id'],
            name=ds_data['name'],
            source=ds_data.get('source', 'unknown'),
            description=ds_data.get('description'),
            num_samples=ds_data.get('num_samples'),
            num_classes=ds_data.get('num_classes'),
            splits=ds_data.get('splits', []),
            metadata=ds_data.get('metadata', {})
        )

        logger.info(f"Retrieved dataset: {dataset}")
        return dataset

    def download_dataset_data(
        self,
        dataset_id: str,
        split: str = 'train',
        format: str = 'numpy'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Download dataset data from backend.

        Args:
            dataset_id: UUID of the dataset
            split: Dataset split ('train', 'val', 'test')
            format: Data format ('numpy', 'json', 'csv')

        Returns:
            Tuple of (x_data, y_data) as numpy arrays

        Raises:
            requests.HTTPError: If dataset or split not found
            ValueError: If format is not supported

        Example:
            >>> x_train, y_train = client.download_dataset_data(
            ...     dataset_id="uuid-123",
            ...     split="train"
            ... )
            >>> print(f"Downloaded {len(x_train)} training samples")
        """
        if format not in ('numpy', 'json', 'csv'):
            raise ValueError(f"Unsupported format: {format}. Use 'numpy', 'json', or 'csv'")

        params = {'split': split, 'format': format}
        response = self._request('GET', f'/datasets/{dataset_id}/download', params=params)

        if format == 'numpy':
            # Backend returns NPZ file as bytes
            npz_bytes = io.BytesIO(response.content)
            with np.load(npz_bytes, allow_pickle=True) as data:
                x_data = data['x']
                y_data = data['y']

            logger.info(
                f"Downloaded {split} split for dataset {dataset_id}: "
                f"{len(x_data)} samples"
            )
            return x_data, y_data

        elif format == 'json':
            # Backend returns JSON with data and labels
            data = response.json()
            x_data = np.array(data['x'])
            y_data = np.array(data['y'])
            return x_data, y_data

        else:  # CSV
            # Backend returns CSV, need to parse
            raise NotImplementedError("CSV format support not yet implemented")

    def import_huggingface_dataset(
        self,
        dataset_name: str,
        dataset_config: Optional[str] = None,
        split_mapping: Optional[Dict[str, str]] = None
    ) -> BackendDatasetInfo:
        """
        Import a dataset from HuggingFace Hub via backend.

        Args:
            dataset_name: HuggingFace dataset name (e.g., 'fashion_mnist')
            dataset_config: Optional dataset configuration/subset
            split_mapping: Optional mapping of HF splits to standard splits
                          (e.g., {'train': 'train', 'test': 'test'})

        Returns:
            BackendDatasetInfo for the imported dataset

        Raises:
            requests.HTTPError: If import fails

        Example:
            >>> dataset = client.import_huggingface_dataset(
            ...     dataset_name='fashion_mnist',
            ...     split_mapping={'train': 'train', 'test': 'test'}
            ... )
            >>> print(f"Imported {dataset.name} with ID {dataset.id}")
        """
        payload = {
            'dataset_name': dataset_name
        }
        if dataset_config:
            payload['dataset_config'] = dataset_config
        if split_mapping:
            payload['split_mapping'] = split_mapping

        response = self._request('POST', '/datasets/huggingface/import', json=payload)
        ds_data = response.json()

        dataset = BackendDatasetInfo(
            id=ds_data['id'],
            name=ds_data['name'],
            source='huggingface',
            description=ds_data.get('description'),
            num_samples=ds_data.get('num_samples'),
            num_classes=ds_data.get('num_classes'),
            splits=ds_data.get('splits', []),
            metadata=ds_data.get('metadata', {})
        )

        logger.info(f"Imported HuggingFace dataset: {dataset}")
        return dataset

    def delete_dataset(self, dataset_id: str) -> bool:
        """
        Delete a dataset from backend.

        Args:
            dataset_id: UUID of the dataset to delete

        Returns:
            True if deletion was successful

        Raises:
            requests.HTTPError: If dataset not found or deletion fails

        Example:
            >>> success = client.delete_dataset("uuid-123")
            >>> print(f"Deletion successful: {success}")
        """
        self._request('DELETE', f'/datasets/{dataset_id}')
        logger.info(f"Deleted dataset {dataset_id}")
        return True

    def get_dataset_statistics(self, dataset_id: str) -> Dict[str, Any]:
        """
        Get statistical information about a dataset.

        Args:
            dataset_id: UUID of the dataset

        Returns:
            Dictionary with statistics (mean, std, class distribution, etc.)

        Example:
            >>> stats = client.get_dataset_statistics("uuid-123")
            >>> print(f"Mean: {stats['mean']}, Std: {stats['std']}")
        """
        response = self._request('GET', f'/datasets/{dataset_id}/statistics')
        stats = response.json()

        logger.info(f"Retrieved statistics for dataset {dataset_id}")
        return stats

    def create_augmentation_pipeline(
        self,
        dataset_id: str,
        transformations: List[Dict[str, Any]]
    ) -> str:
        """
        Create a data augmentation pipeline for a dataset.

        Args:
            dataset_id: UUID of the dataset
            transformations: List of Albumentations transformations
                           (e.g., [{'type': 'HorizontalFlip', 'p': 0.5}])

        Returns:
            Pipeline ID (UUID)

        Example:
            >>> pipeline_id = client.create_augmentation_pipeline(
            ...     dataset_id="uuid-123",
            ...     transformations=[
            ...         {'type': 'HorizontalFlip', 'p': 0.5},
            ...         {'type': 'RandomRotate90', 'p': 0.5}
            ...     ]
            ... )
            >>> print(f"Created pipeline: {pipeline_id}")
        """
        payload = {
            'dataset_id': dataset_id,
            'transformations': transformations
        }

        response = self._request('POST', '/datasets/augmentation/pipeline', json=payload)
        result = response.json()
        pipeline_id = result['pipeline_id']

        logger.info(f"Created augmentation pipeline {pipeline_id} for dataset {dataset_id}")
        return pipeline_id

    def health_check(self) -> bool:
        """
        Check if backend API is healthy and accessible.

        Returns:
            True if backend is healthy, False otherwise

        Example:
            >>> if client.health_check():
            ...     print("Backend is healthy")
            ... else:
            ...     print("Backend is down")
        """
        try:
            response = self._request('GET', '/health')
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def close(self):
        """
        Close the client session.

        Should be called when done using the client to clean up resources.

        Example:
            >>> client.close()
        """
        self.session.close()
        logger.info("Closed BackendAPIClient session")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        """String representation."""
        auth = "authenticated" if (self.api_key or self.jwt_token) else "unauthenticated"
        return f"BackendAPIClient(base_url='{self.base_url}', {auth})"


# Convenience function for quick client creation
def create_backend_client(
    base_url: str = "http://localhost:8000",
    api_key: Optional[str] = None,
    jwt_token: Optional[str] = None
) -> BackendAPIClient:
    """
    Convenience function to create a BackendAPIClient.

    Args:
        base_url: Backend API base URL
        api_key: Optional API key for authentication
        jwt_token: Optional JWT token for authentication

    Returns:
        BackendAPIClient instance

    Example:
        >>> client = create_backend_client(
        ...     base_url="http://localhost:8000",
        ...     api_key="your_api_key"
        ... )
        >>> datasets = client.list_datasets()
    """
    return BackendAPIClient(
        base_url=base_url,
        api_key=api_key,
        jwt_token=jwt_token
    )
