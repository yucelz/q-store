# Q-Store Quick Start Guide

Get up and running with Q-Store in 5 minutes!

## Prerequisites

- Python 3.11+
- Conda package manager
- Pinecone account (free tier available)

## Step 1: Installation

```bash
# Clone the repository
git clone https://github.com/yucelz/q-store.git
cd q-store

# Create and activate conda environment
conda env create -f environment.yml
conda activate q-store

# Install the package
pip install -e .

# Install Pinecone (required)
pip install pinecone
```

## Step 2: Get API Keys

### Pinecone (Required)
1. Sign up at [pinecone.io](https://www.pinecone.io/)
2. Create a free account
3. Go to your dashboard and copy your API key
4. Note your environment (e.g., `us-east-1`)

### IonQ (Optional - for quantum features)
1. Sign up at [cloud.ionq.com](https://cloud.ionq.com/)
2. Go to Settings → API Keys
3. Copy your API key

## Step 3: Configure Environment

Create a `.env` file in the project root:

```bash
# Create .env file
cat > .env << 'EOF'
# Required: Pinecone credentials
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=us-east-1

# Optional: IonQ for quantum features
IONQ_API_KEY=your_ionq_api_key_here
EOF
```

**Important:** Replace `your_pinecone_api_key_here` with your actual API key!

## Step 4: Run Your First Test

**Verify your installation:**

```bash
python verify_installation.py
```

You should see:
```
============================================================
Q-Store Installation Verification
============================================================

Checking imports...
  ✓ NumPy
  ✓ SciPy
  ✓ Cirq
  ✓ Pinecone
  ✓ Q-Store

Checking .env file...
  ✓ .env file exists
  ✓ PINECONE_API_KEY set
  ✓ PINECONE_ENVIRONMENT set

✓ All checks passed!
```

**Run the full demo:**

```bash
python examples/quantum_db_quickstart.py
```

You should see output like:

```
============================================================
QUANTUM DATABASE - INTERACTIVE DEMO
============================================================

=== Quantum Database Setup ===

Configuration:
  - Pinecone Index: quantum-demo
  - Pinecone Environment: us-east-1
  - Dimension: 768
  - Quantum Enabled: True
  - Superposition: True

Initializing database...
INFO:q_store.quantum_database:Pinecone initialized with environment: us-east-1
INFO:q_store.quantum_database:Creating Pinecone index: quantum-demo
✓ Database initialized successfully

=== Example 1: Basic Operations ===
Creating sample embeddings...
Inserting documents...
  ✓ Inserted doc_1
  ✓ Inserted doc_2
  ✓ Inserted doc_3
...
```

## Step 5: Your First Program

Create `my_first_qstore.py`:

```python
import asyncio
import numpy as np
from dotenv import load_dotenv
import os
from q_store import QuantumDatabase, DatabaseConfig

# Load environment variables from .env
load_dotenv()

async def main():
    # Configure database
    config = DatabaseConfig(
        pinecone_api_key=os.getenv('PINECONE_API_KEY'),
        pinecone_environment=os.getenv('PINECONE_ENVIRONMENT', 'us-east-1'),
        pinecone_index_name='my-first-index',
        pinecone_dimension=128,  # Small dimension for testing
        ionq_api_key=os.getenv('IONQ_API_KEY'),  # Optional
    )
    
    # Initialize database
    db = QuantumDatabase(config)
    
    async with db.connect():
        print("✓ Connected to Q-Store!")
        
        # Insert a vector
        embedding = np.random.rand(128)
        await db.insert(
            id='test_doc_1',
            vector=embedding,
            metadata={'type': 'test', 'timestamp': '2025-01-01'}
        )
        print("✓ Inserted vector")
        
        # Query similar vectors
        results = await db.query(
            vector=embedding,
            top_k=5
        )
        print(f"✓ Found {len(results)} similar vectors")
        
        # Display results
        for i, result in enumerate(results, 1):
            print(f"  {i}. ID: {result.id}, Score: {result.score:.4f}")

if __name__ == '__main__':
    asyncio.run(main())
```

Run it:

```bash
python my_first_qstore.py
```

## What's Next?

- 📖 Read the full [README.md](README.md) for detailed documentation
- 💡 Explore [examples/](examples/) for more use cases
- 🏗️ Review [quantum_db_design_v2.md](quantum_db_design_v2.md) for architecture
- 🧪 Run tests: `pytest tests/ -v`

## Common Commands

```bash
# Activate environment
conda activate q-store

# Run quickstart demo
python examples/quantum_db_quickstart.py

# Run tests
pytest tests/ -v

# Update dependencies
conda env update -f environment.yml

# Deactivate environment
conda deactivate
```

## Troubleshooting

### Error: "No module named 'q_store'"
```bash
pip install -e .
```

### Error: "Pinecone package is required"
```bash
pip uninstall -y pinecone-client
pip install pinecone
```

### Error: "PINECONE_API_KEY not found"
Make sure your `.env` file exists in the project root and contains valid API keys.

### Need Help?
- GitHub Issues: [github.com/yucelz/q-store/issues](https://github.com/yucelz/q-store/issues)
- Email: yucelz@gmail.com

---

**Happy Quantum Computing! 🚀**
