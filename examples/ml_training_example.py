"""
Machine Learning Training Example
Demonstrates quantum database for training data selection and optimization.
"""

import os
import numpy as np
from getpass import getpass
from dotenv import load_dotenv
from q_store import QuantumDatabase, QuantumDatabaseConfig

# Load environment variables from .env file
load_dotenv()


def generate_training_sample(label: int, noise: float = 0.1) -> np.ndarray:
    """Generate synthetic training sample"""
    base = np.random.randn(64)
    return base + np.random.randn(64) * noise


def main():
    """ML training quantum database example"""
    
    print("=== Q-Store: ML Training Example ===\n")
    
    # Get API key
    api_key = os.getenv('IONQ_API_KEY') or getpass('Enter your IonQ API key: ')
    
    # Initialize database
    db = QuantumDatabase(
        ionq_api_key=api_key,
        target_device='simulator',
        enable_superposition=True,
        enable_tunneling=True,
        default_coherence_time=3000
    )
    
    print("✓ Initialized ML training database\n")
    
    # 1. Store training examples with multiple task contexts
    print("1. Storing training examples with multi-task contexts...")
    
    n_samples = 20
    
    for i in range(n_samples):
        label = i % 3
        sample = generate_training_sample(label)
        
        # Each sample can be used for multiple tasks (superposition)
        db.insert(
            id=f'sample_{i}',
            vector=sample,
            contexts=[
                ('classification', 0.6),
                ('regression', 0.3),
                ('clustering', 0.1)
            ],
            metadata={
                'label': label,
                'difficulty': np.random.choice(['easy', 'medium', 'hard'])
            }
        )
    
    print(f"  ✓ Stored {n_samples} training samples\n")
    
    # 2. Context-aware batch sampling
    print("2. Sampling training batch for classification task...")
    
    # Model state (simplified)
    model_state = np.random.randn(64)
    
    # Get samples relevant to classification task
    classification_batch = db.query(
        vector=model_state,
        context='classification',  # Collapses to classification context
        mode='exploratory',  # Broad coverage for diversity
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
    hard_negatives = db.query(
        vector=model_state,
        context='classification',
        enable_tunneling=True,  # Find hard examples
        mode='precise',
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
        config = np.random.randn(64)
        db.insert(
            id=f'config_{i}',
            vector=config,
            metadata={
                'learning_rate': 10 ** np.random.uniform(-4, -2),
                'batch_size': np.random.choice([16, 32, 64, 128])
            }
        )
        configs.append(config)
    
    # Target: best performance state
    target_performance = np.random.randn(64)
    
    # Use tunneling to escape local optima
    best_configs = db.query(
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
    informative_samples = db.query(
        vector=uncertain_region,
        context='classification',
        mode='balanced',
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
    easy_batch = [
        r for r in db.query(model_state, context='classification', top_k=20)
        if r.metadata.get('difficulty') == 'easy'
    ]
    
    print(f"  Early training: {len(easy_batch)} easy samples")
    
    # Later training: harder samples
    hard_batch = [
        r for r in db.query(model_state, context='classification', top_k=20)
        if r.metadata.get('difficulty') == 'hard'
    ]
    
    print(f"  Later training: {len(hard_batch)} hard samples\n")
    
    # 7. Regime discovery with tunneling engine
    print("7. Discovering training regimes...")
    
    # Get all training samples
    all_samples = [v for k, v in db.classical_store.items() if 'sample' in k]
    
    if len(all_samples) > 0:
        regimes = db.tunneling_engine.discover_regimes(
            historical_data=all_samples,
            n_regimes=3
        )
        
        print(f"  Discovered {len(regimes)} training regimes:")
        for i, regime in enumerate(regimes):
            print(f"    - Regime {i+1}: {len(regime)} samples")
    print()
    
    # Stats
    stats = db.get_stats()
    print(f"Total training samples: {stats['total_vectors']}")
    print(f"Active quantum states: {stats['quantum_states']}")
    
    print("\n=== ML training example completed! ===")


if __name__ == '__main__':
    main()
