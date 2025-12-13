"""
Financial Services Example
Demonstrates quantum database for portfolio correlation and crisis detection.
"""

import os
import numpy as np
from getpass import getpass
from dotenv import load_dotenv
from q_store import QuantumDatabase, DatabaseConfig

# Load environment variables from .env file
load_dotenv()


def generate_stock_embedding(volatility: float = 0.1) -> np.ndarray:
    """Generate synthetic stock embedding"""
    return np.random.randn(128) * volatility


def main():
    """Financial services quantum database example"""
    
    print("=== Q-Store: Financial Services Example ===\n")
    
    # Get API key
    api_key = os.getenv('IONQ_API_KEY') or getpass('Enter your IonQ API key: ')
    
    # Initialize database
    db = QuantumDatabase(
        ionq_api_key=api_key,
        target_device='simulator',
        enable_entanglement=True,
        enable_tunneling=True,
        correlation_threshold=0.85
    )
    
    print("✓ Initialized financial quantum database\n")
    
    # 1. Store stock embeddings with market contexts
    print("1. Storing stock portfolio with market contexts...")
    
    stocks = {
        'AAPL': generate_stock_embedding(0.08),
        'MSFT': generate_stock_embedding(0.07),
        'GOOGL': generate_stock_embedding(0.09),
        'AMZN': generate_stock_embedding(0.11),
        'META': generate_stock_embedding(0.15)
    }
    
    for ticker, embedding in stocks.items():
        db.insert(
            id=ticker,
            vector=embedding,
            contexts=[
                ('bull_market', 0.4),
                ('normal_market', 0.4),
                ('bear_market', 0.2)
            ],
            coherence_time=5000,  # 5 seconds (recent data)
            metadata={'sector': 'Tech', 'type': 'stock'}
        )
    
    print(f"  ✓ Stored {len(stocks)} stocks with market contexts\n")
    
    # 2. Create entangled groups (sector correlations)
    print("2. Creating entangled sector groups...")
    
    db.create_entangled_group(
        group_id='tech_sector',
        entity_ids=['AAPL', 'MSFT', 'GOOGL'],
        correlation_strength=0.85
    )
    
    db.create_entangled_group(
        group_id='mega_caps',
        entity_ids=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
        correlation_strength=0.75
    )
    
    print("  ✓ Created tech_sector group (0.85 correlation)")
    print("  ✓ Created mega_caps group (0.75 correlation)\n")
    
    # 3. Query for similar stocks in different market contexts
    print("3. Finding similar stocks in different market contexts...")
    
    query_portfolio = np.mean([stocks['AAPL'], stocks['MSFT']], axis=0)
    
    # Bull market context
    bull_results = db.query(
        vector=query_portfolio,
        context='bull_market',
        top_k=3
    )
    
    print("  Bull market context:")
    for result in bull_results:
        print(f"    - {result.id} (score: {result.score:.4f})")
    
    # Bear market context
    bear_results = db.query(
        vector=query_portfolio,
        context='bear_market',
        top_k=3
    )
    
    print("\n  Bear market context:")
    for result in bear_results:
        print(f"    - {result.id} (score: {result.score:.4f})")
    print()
    
    # 4. Crisis detection via quantum tunneling
    print("4. Crisis pattern detection with tunneling...")
    
    # Simulate historical crisis patterns
    crisis_states = [
        generate_stock_embedding(0.3) for _ in range(5)  # High volatility
    ]
    
    # Store historical crises
    for i, state in enumerate(crisis_states):
        db.insert(
            id=f'crisis_{i}',
            vector=state,
            metadata={'type': 'crisis', 'year': 2000 + i * 5}
        )
    
    # Query with tunneling to find crisis precursors
    current_state = generate_stock_embedding(0.12)
    
    crisis_results = db.query(
        vector=current_state,
        enable_tunneling=True,  # Find distant crisis patterns
        top_k=3
    )
    
    print("  Crisis patterns found:")
    for result in crisis_results:
        if 'crisis' in result.id:
            year = result.metadata.get('year', 'unknown')
            print(f"    - Similar to {year} crisis (score: {result.score:.4f})")
    print()
    
    # 5. Correlation analysis
    print("5. Analyzing correlations...")
    
    correlation = db.entanglement_registry.measure_correlation('AAPL', 'MSFT')
    print(f"  AAPL ↔ MSFT correlation: {correlation}")
    
    correlation = db.entanglement_registry.measure_correlation('AAPL', 'META')
    print(f"  AAPL ↔ META correlation: {correlation}")
    
    # Get all correlated stocks
    partners = db.entanglement_registry.get_entangled_partners('AAPL')
    print(f"  AAPL entangled with: {partners}\n")
    
    # 6. Portfolio update with automatic propagation
    print("6. Updating portfolio (entanglement propagation)...")
    
    new_aapl_state = generate_stock_embedding(0.09)
    affected = db.entanglement_registry.update_entity('AAPL', new_aapl_state)
    
    print(f"  ✓ Updated AAPL")
    print(f"  ✓ Auto-propagated to: {affected}\n")
    
    # Stats
    stats = db.get_stats()
    print(f"Total assets tracked: {stats['total_vectors']}")
    print(f"Active quantum states: {stats['quantum_states']}")
    print(f"Correlation groups: {stats['entangled_groups']}")
    
    print("\n=== Financial example completed! ===")


if __name__ == '__main__':
    main()
