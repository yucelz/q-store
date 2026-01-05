"""
Backend API Client Example - Demonstrates Q-Store Backend API integration.

This example shows how to:
- Connect to the Q-Store Backend API
- List and retrieve datasets
- Download dataset data
- Import from HuggingFace
- Integrate with Label Studio
- Apply augmentation
"""

import numpy as np
from q_store.data.backend_client import (
    BackendAPIClient,
    BackendConfig
)


def example_basic_connection():
    """Connect to backend API."""
    print("\n" + "="*70)
    print("Example 1: Basic API Connection")
    print("="*70)

    config = BackendConfig(
        base_url='http://localhost:8000',
        api_key='your_api_key_here',
        timeout=30
    )

    try:
        client = BackendAPIClient(config)

        # Test connection
        health = client.health_check()
        print(f"Connection: {health.get('status', 'unknown')}")
        print(f"Version: {health.get('version', 'unknown')}")
        print("✓ Successfully connected to backend")

        return client
    except Exception as e:
        print(f"⚠ Backend not available: {e}")
        print("Make sure the Q-Store Backend is running at http://localhost:8000")
        return None


def example_list_datasets(client):
    """List available datasets."""
    print("\n" + "="*70)
    print("Example 2: List Datasets")
    print("="*70)

    if client is None:
        print("⚠ Client not available")
        return

    try:
        datasets = client.list_datasets(limit=10, offset=0)

        print(f"Found {len(datasets)} datasets:")
        for ds in datasets[:5]:  # Show first 5
            print(f"  - {ds['name']} (ID: {ds['id'][:8]}...)")
            print(f"    Source: {ds.get('source', 'unknown')}")
            print(f"    Samples: {ds.get('num_samples', 'unknown')}")

        print("✓ Dataset listing successful")

        return datasets
    except Exception as e:
        print(f"⚠ Failed to list datasets: {e}")
        return []


def example_get_dataset_details(client, dataset_id):
    """Get detailed dataset information."""
    print("\n" + "="*70)
    print("Example 3: Get Dataset Details")
    print("="*70)

    if client is None:
        print("⚠ Client not available")
        return

    try:
        dataset = client.get_dataset(dataset_id)

        print(f"Dataset: {dataset['name']}")
        print(f"ID: {dataset['id']}")
        print(f"Source: {dataset.get('source', 'unknown')}")
        print(f"Samples: {dataset.get('num_samples', 'unknown')}")
        print(f"Classes: {dataset.get('num_classes', 'unknown')}")
        print(f"Features: {dataset.get('num_features', 'unknown')}")
        print(f"Created: {dataset.get('created_at', 'unknown')}")
        print("✓ Dataset details retrieved")

        return dataset
    except Exception as e:
        print(f"⚠ Failed to get dataset: {e}")
        return None


def example_download_dataset(client, dataset_id):
    """Download dataset data."""
    print("\n" + "="*70)
    print("Example 4: Download Dataset Data")
    print("="*70)

    if client is None:
        print("⚠ Client not available")
        return

    try:
        # Download train split
        train_data = client.download_dataset_data(
            dataset_id=dataset_id,
            split='train'
        )

        print(f"Downloaded train data:")
        print(f"  X shape: {train_data['x'].shape}")
        print(f"  Y shape: {train_data['y'].shape}")

        # Download test split
        test_data = client.download_dataset_data(
            dataset_id=dataset_id,
            split='test'
        )

        print(f"Downloaded test data:")
        print(f"  X shape: {test_data['x'].shape}")
        print(f"  Y shape: {test_data['y'].shape}")

        print("✓ Dataset download successful")

        return train_data, test_data
    except Exception as e:
        print(f"⚠ Failed to download dataset: {e}")
        return None, None


def example_import_from_huggingface(client):
    """Import dataset from HuggingFace."""
    print("\n" + "="*70)
    print("Example 5: Import from HuggingFace")
    print("="*70)

    if client is None:
        print("⚠ Client not available")
        return

    try:
        result = client.import_from_huggingface(
            dataset_name='mnist',
            config_name=None,
            split='train',
            feature_column='image',
            label_column='label'
        )

        print(f"Import initiated:")
        print(f"  Dataset ID: {result['dataset_id']}")
        print(f"  Status: {result['status']}")
        print(f"  Message: {result.get('message', 'Import in progress')}")
        print("✓ HuggingFace import successful")

        return result['dataset_id']
    except Exception as e:
        print(f"⚠ Failed to import from HuggingFace: {e}")
        return None


def example_label_studio_integration(client):
    """Integrate with Label Studio."""
    print("\n" + "="*70)
    print("Example 6: Label Studio Integration")
    print("="*70)

    if client is None:
        print("⚠ Client not available")
        return

    try:
        # Create Label Studio project
        project = client.create_label_studio_project(
            project_name='quantum_classification',
            label_config='''
            <View>
              <Image name="image" value="$image"/>
              <Choices name="class" toName="image">
                <Choice value="class_0"/>
                <Choice value="class_1"/>
                <Choice value="class_2"/>
              </Choices>
            </View>
            '''
        )

        print(f"Created Label Studio project:")
        print(f"  Project ID: {project['id']}")
        print(f"  Name: {project['name']}")
        print(f"  URL: {project['url']}")

        # Export labeled data
        labeled_data = client.export_labeled_data(
            project_id=project['id'],
            export_format='json'
        )

        print(f"Exported {len(labeled_data)} labeled samples")
        print("✓ Label Studio integration successful")

        return project
    except Exception as e:
        print(f"⚠ Label Studio integration failed: {e}")
        return None


def example_apply_augmentation(client, dataset_id):
    """Apply augmentation to dataset."""
    print("\n" + "="*70)
    print("Example 7: Apply Augmentation")
    print("="*70)

    if client is None:
        print("⚠ Client not available")
        return

    try:
        result = client.apply_augmentation(
            dataset_id=dataset_id,
            transforms=[
                {'type': 'horizontal_flip', 'p': 0.5},
                {'type': 'rotate', 'limit': 15, 'p': 0.3},
                {'type': 'random_brightness_contrast', 'p': 0.5}
            ],
            num_augmented_per_sample=2
        )

        print(f"Augmentation applied:")
        print(f"  Original samples: {result['original_count']}")
        print(f"  Augmented samples: {result['augmented_count']}")
        print(f"  Total samples: {result['total_count']}")
        print(f"  New dataset ID: {result['new_dataset_id']}")
        print("✓ Augmentation successful")

        return result['new_dataset_id']
    except Exception as e:
        print(f"⚠ Augmentation failed: {e}")
        return None


def example_upload_dataset(client):
    """Upload custom dataset."""
    print("\n" + "="*70)
    print("Example 8: Upload Custom Dataset")
    print("="*70)

    if client is None:
        print("⚠ Client not available")
        return

    try:
        # Create sample dataset
        x_train = np.random.rand(100, 28, 28)
        y_train = np.random.randint(0, 10, 100)
        x_test = np.random.rand(20, 28, 28)
        y_test = np.random.randint(0, 10, 20)

        result = client.upload_dataset(
            name='my_custom_dataset',
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            metadata={
                'description': 'Custom quantum ML dataset',
                'num_classes': 10,
                'image_size': (28, 28)
            }
        )

        print(f"Dataset uploaded:")
        print(f"  Dataset ID: {result['dataset_id']}")
        print(f"  Name: {result['name']}")
        print(f"  Train samples: {len(x_train)}")
        print(f"  Test samples: {len(x_test)}")
        print("✓ Upload successful")

        return result['dataset_id']
    except Exception as e:
        print(f"⚠ Upload failed: {e}")
        return None


def example_complete_workflow():
    """Complete workflow: upload, augment, download."""
    print("\n" + "="*70)
    print("Example 9: Complete Backend Workflow")
    print("="*70)

    # Step 1: Connect
    print("\nStep 1: Connect to backend")
    config = BackendConfig(
        base_url='http://localhost:8000',
        api_key='your_api_key_here'
    )

    try:
        client = BackendAPIClient(config)
        print("  ✓ Connected")
    except Exception as e:
        print(f"  ⚠ Connection failed: {e}")
        return

    # Step 2: Upload dataset
    print("\nStep 2: Upload dataset")
    try:
        x_data = np.random.rand(100, 28, 28)
        y_data = np.random.randint(0, 3, 100)

        upload_result = client.upload_dataset(
            name='workflow_dataset',
            x_train=x_data[:80],
            y_train=y_data[:80],
            x_test=x_data[80:],
            y_test=y_data[80:]
        )
        dataset_id = upload_result['dataset_id']
        print(f"  ✓ Dataset uploaded: {dataset_id[:8]}...")
    except Exception as e:
        print(f"  ⚠ Upload failed: {e}")
        return

    # Step 3: Apply augmentation
    print("\nStep 3: Apply augmentation")
    try:
        aug_result = client.apply_augmentation(
            dataset_id=dataset_id,
            transforms=[
                {'type': 'horizontal_flip', 'p': 0.5},
                {'type': 'rotate', 'limit': 10, 'p': 0.3}
            ],
            num_augmented_per_sample=1
        )
        aug_dataset_id = aug_result['new_dataset_id']
        print(f"  ✓ Augmented dataset: {aug_dataset_id[:8]}...")
        print(f"  Original: {aug_result['original_count']} → Augmented: {aug_result['total_count']}")
    except Exception as e:
        print(f"  ⚠ Augmentation failed: {e}")
        return

    # Step 4: Download augmented data
    print("\nStep 4: Download augmented data")
    try:
        train_data = client.download_dataset_data(
            dataset_id=aug_dataset_id,
            split='train'
        )
        print(f"  ✓ Downloaded: {train_data['x'].shape}")
    except Exception as e:
        print(f"  ⚠ Download failed: {e}")
        return

    print("\n✓ Complete workflow successful!")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("Q-Store Backend API Client Examples")
    print("="*70)
    print("\nNOTE: These examples require the Q-Store Backend to be running.")
    print("Start it with: python -m q_store.backend.main")
    print("="*70)

    # Example 1: Basic connection
    client = example_basic_connection()

    if client is None:
        print("\n⚠ Backend not available. Showing example code only.")
        print("\nTo run these examples:")
        print("1. Start the Q-Store Backend: python -m q_store.backend.main")
        print("2. Configure your API key in the examples")
        print("3. Run this script again")
        return

    # Example 2: List datasets
    datasets = example_list_datasets(client)

    # Use first dataset for remaining examples
    if datasets:
        dataset_id = datasets[0]['id']

        # Example 3: Get details
        example_get_dataset_details(client, dataset_id)

        # Example 4: Download data
        example_download_dataset(client, dataset_id)

        # Example 7: Augmentation
        example_apply_augmentation(client, dataset_id)

    # Example 5: HuggingFace import
    example_import_from_huggingface(client)

    # Example 6: Label Studio
    example_label_studio_integration(client)

    # Example 8: Upload
    example_upload_dataset(client)

    # Example 9: Complete workflow
    example_complete_workflow()

    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)


if __name__ == '__main__':
    main()
