"""
Machine Learning Training Example
Demonstrates quantum database for training data selection and optimization.
"""

import asyncio
import os
import numpy as np
from getpass import getpass
from dotenv import load_dotenv
from q_store import QuantumDatabase, DatabaseConfig, QueryMode

# Load environment variables from .env file
load_dotenv()


def generate_training_sample(label: int, noise: float = 0.1) -> np.ndarray:
    """Generate synthetic training sample"""
    base = np.random.randn(64)
    return base + np.random.randn(64) * noise


async def main():
    """ML training quantum database example"""
    
    print("=== Q-Store: ML Training Example ===\n")
    
    # Get API keys
    pinecone_key = os.getenv('PINECONE_API_KEY')
    ionq_key = os.getenv('IONQ_API_KEY')
    pinecone_environment = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1')
    
    if not pinecone_key:
        raise ValueError(
            "PINECONE_API_KEY is required. Please add it to your .env file.\n"
            "Get your API key from: https://www.pinecone.io/"
        )
    
    # Create configuration
    config = DatabaseConfig(
        # Pinecone settings
        pinecone_api_key=pinecone_key,
        pinecone_environment=pinecone_environment,
        pinecone_index_name="ml-training-demo",
        pinecone_dimension=64,  # Match our sample dimension
        pinecone_metric="cosine",
        
        # IonQ quantum settings (optional)
        ionq_api_key=ionq_key,
        ionq_target="simulator",
        
        # Feature flags
        enable_quantum=True if ionq_key else False,
        enable_superposition=True if ionq_key else False,
        enable_entanglement=True,
        enable_tunneling=True if ionq_key else False,
        
        # Performance tuning
        default_coherence_time=3000.0,
        max_quantum_states=500,
        classical_candidate_pool=1000
    )
    
    # Initialize database
    db = QuantumDatabase(config=config)
    await db.initialize()
    
    print("✓ Initialized ML training database\n")
    
    # 1. Store training examples with multiple task contexts
    print("1. Storing training examples with multi-task contexts...")
    
    n_samples = 20
    
    for i in range(n_samples):
        label = i % 3
        sample = generate_training_sample(label)
        
        # Each sample can be used for multiple tasks (superposition)
        await db.insert(
            id=f'sample_{i}',
            vector=sample,
            contexts=[
                ('classification', 0.6),
                ('regression', 0.3),
                ('clustering', 0.1)
            ],
            metadata={
                'label': int(label),
                'difficulty': str(np.random.choice(['easy', 'medium', 'hard']))
            }
        )
    
    print(f"  ✓ Stored {n_samples} training samples\n")
    
    # 2. Context-aware batch sampling
    print("2. Sampling training batch for classification task...")
    
    # Model state (simplified)
    model_state = np.random.randn(64)
    
    # Get samples relevant to classification task
    classification_batch = await db.query(
        vector=model_state,
        context='classification',  # Collapses to classification context
        mode=QueryMode.EXPLORATORY,  # Broad coverage for diversity
        top_k=8
    )
    
    print(f"  Sampled {len(classification_batch)} examples:")
    for result in classification_batch[:5]:
        difficulty = result.metadata.get('difficulty', 'unknown')
        print(f"    - {result.id} (difficulty: {difficulty}, score: {result.score:.4f})")
    print()
    
    # 3. Hard negative mining with tunneling
    print("3. Hard negative mining using quantum tunneling...")
    
    # Find challenging examples (distant but relevant)
    hard_negatives = await db.query(
        vector=model_state,
        context='classification',
        enable_tunneling=True,  # Find hard examples
        mode=QueryMode.PRECISE,
        top_k=5
    )
    
    print("  Hard negative examples:")
    for result in hard_negatives:
        print(f"    - {result.id} (score: {result.score:.4f})")
    print()
    
    # 4. Hyperparameter optimization with tunneling
    print("4. Hyperparameter search using quantum tunneling...")
    
    # Simulate hyperparameter configurations
    configs = []
    for i in range(10):
        # Each config represented as vector
        config_vec = np.random.randn(64)
        await db.insert(
            id=f'config_{i}',
            vector=config_vec,
            metadata={
                'learning_rate': float(10 ** np.random.uniform(-4, -2)),
                'batch_size': int(np.random.choice([16, 32, 64, 128]))
            }
        )
        configs.append(config_vec)
    
    # Target: best performance state
    target_performance = np.random.randn(64)
    
    # Use tunneling to escape local optima
    best_configs = await db.query(
        vector=target_performance,
        enable_tunneling=True,
        top_k=3
    )
    
    print("  Top configurations found:")
    for result in best_configs:
        if 'config' in result.id:
            lr = result.metadata.get('learning_rate', 0)
            bs = result.metadata.get('batch_size', 0)
            print(f"    - {result.id}: lr={lr:.2e}, batch_size={bs}")
    print()
    
    # 5. Active learning: query most informative samples
    print("5. Active learning: selecting informative samples...")
    
    # Model uncertainty representation
    uncertain_region = np.random.randn(64)
    
    # Find samples in uncertain region
    informative_samples = await db.query(
        vector=uncertain_region,
        context='classification',
        mode=QueryMode.BALANCED,
        top_k=5
    )
    
    print("  Most informative samples to label:")
    for result in informative_samples:
        if 'sample' in result.id:
            print(f"    - {result.id} (informativeness: {result.score:.4f})")
    print()
    
    # 6. Curriculum learning: progressive difficulty
    print("6. Curriculum learning: adaptive difficulty...")
    
    # Early training: easy samples
    easy_results = await db.query(model_state, context='classification', top_k=20)
    easy_batch = [
        r for r in easy_results
        if r.metadata.get('difficulty') == 'easy'
    ]
    
    print(f"  Early training: {len(easy_batch)} easy samples")
    
    # Later training: harder samples
    hard_results = await db.query(model_state, context='classification', top_k=20)
    hard_batch = [
        r for r in hard_results
        if r.metadata.get('difficulty') == 'hard'
    ]
    
    print(f"  Later training: {len(hard_batch)} hard samples\n")
    
    print("\n=== ML training example completed! ===")
    
    # Cleanup
    await db.close()


if __name__ == '__main__':
    asyncio.run(main())
