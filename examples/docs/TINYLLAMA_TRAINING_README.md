# TinyLlama React Fine-Tuning with Q-Store Quantum Database

This example demonstrates how to use Q-Store's quantum-enhanced database for intelligent training data management when fine-tuning TinyLlama on React code generation tasks.

## Overview

Traditional ML training samples data randomly or sequentially. This example shows how quantum computing concepts can enhance training through:

1. **Quantum Superposition**: Store training samples in multiple contexts simultaneously (generation, debugging, explanation, etc.)
2. **Quantum Entanglement**: Group related samples that update together based on correlation
3. **Curriculum Learning**: Progressively sample from easy → medium → hard examples using quantum search
4. **Hard Negative Mining**: Use quantum tunneling to find challenging examples that classical methods miss
5. **Context-Aware Sampling**: Collapse superposition to the most relevant training samples for specific contexts

## Features

### Quantum-Enhanced Training Data Management

- **Multi-Context Storage**: Each training sample exists in superposition across multiple instruction types
- **Intelligent Sampling**: Query-based sample selection using quantum search algorithms
- **Adaptive Difficulty**: Curriculum learning with automatic difficulty estimation
- **Hard Negative Discovery**: Quantum tunneling to find challenging examples
- **Entangled Groups**: Related samples grouped by instruction type for coherent updates

### Traditional ML Training (Optional)

- LoRA fine-tuning of TinyLlama-1.1B
- 4-bit quantization for efficient training
- Customizable hyperparameters
- React code generation focus

## Requirements

### Core Requirements (Q-Store Demo)

```bash
# Q-Store and dependencies
pip install -e ..  # Install q-store from parent directory
pip install numpy python-dotenv

# API Keys (add to .env file)
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=us-east-1
IONQ_API_KEY=your_ionq_key  # Optional, enables quantum features
```

### Optional Requirements (Full Training)

```bash
# For actual model training (requires GPU)
pip install transformers peft datasets torch
pip install accelerate bitsandbytes  # For 4-bit quantization
```

## Quick Start

### 1. Set Up Environment

Create a `.env` file in the q-store root directory:

```env
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=us-east-1
IONQ_API_KEY=your_ionq_api_key_here  # Optional
```

### 2. Run the Demo

```bash
# From the examples directory
cd examples
python tinyllama_react_training.py
```

The script will:
1. Create sample React training data if not found
2. Load data into Q-Store quantum database
3. Demonstrate different quantum-enhanced sampling strategies
4. (Optional) Train TinyLlama model if transformers is installed

### 3. With Your Own Data

Prepare a JSONL file with training samples:

```json
{"instruction": "Create a React counter component", "input": "", "output": "import React..."}
{"instruction": "Explain useState hook", "input": "", "output": "useState is a Hook..."}
{"instruction": "Fix this useEffect bug", "input": "useEffect(() => {...", "output": "Add dependency array..."}
```

Update the configuration:

```python
config = TrainingConfig(
    training_data_file="your_data.jsonl",
    max_samples=1000,
    use_quantum_sampling=True,
    use_curriculum_learning=True
)
```

## How It Works

### 1. Data Loading with Quantum Features

```python
# Each sample is stored with multiple contexts in superposition
await db.insert(
    id='sample_0',
    vector=embedding,
    contexts=[
        ('generation', 0.6),    # 60% generation context
        ('general', 0.3),       # 30% general context
        ('easy', 0.1)          # 10% difficulty context
    ],
    coherence_time=5000.0,     # Relevance decay time
    metadata={'instruction': '...', 'difficulty': 'easy'}
)
```

### 2. Curriculum Learning

```python
# Early epochs: sample easy examples (PRECISE mode)
easy_batch = await data_manager.sample_training_batch(
    batch_size=32,
    epoch=0,
    use_curriculum=True
)

# Later epochs: sample hard examples (EXPLORATORY mode)
hard_batch = await data_manager.sample_training_batch(
    batch_size=32,
    epoch=2,
    use_curriculum=True
)
```

### 3. Context-Specific Sampling

```python
# Focus on specific instruction types
generation_samples = await data_manager.sample_training_batch(
    batch_size=32,
    context='generation',  # Collapses to generation context
    use_curriculum=False
)

debugging_samples = await data_manager.sample_training_batch(
    batch_size=32,
    context='debugging',   # Collapses to debugging context
    use_curriculum=False
)
```

### 4. Hard Negative Mining

```python
# Use quantum tunneling to find challenging examples
hard_negatives = await data_manager.hard_negative_mining(
    model_state=current_model_embedding,
    context='debugging',
    num_negatives=10
)
```

## Configuration Options

### TrainingConfig

```python
@dataclass
class TrainingConfig:
    # Model settings
    model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    output_dir: str = "./tinyllama-react-quantum"
    
    # Data settings
    training_data_file: str = "react_train.jsonl"
    max_samples: int = 1000
    embedding_dim: int = 768
    
    # Training hyperparameters
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    
    # Quantum features
    use_quantum_sampling: bool = True
    use_curriculum_learning: bool = True
    use_hard_negative_mining: bool = True
    coherence_time: float = 5000.0  # ms
```

## Quantum Database Benefits

### 1. Intelligent Sample Selection

Instead of random sampling, the quantum database:
- Finds samples most relevant to current model state
- Balances exploration vs exploitation
- Adapts difficulty based on training progress

### 2. Multi-Context Storage

Each sample can be relevant for multiple tasks:
```
Sample: "Create a React form component"
Contexts:
  - generation (primary)
  - explanation (secondary)
  - medium difficulty
```

### 3. Automatic Grouping

Related samples are entangled:
```
Entangled Group: "debugging_tasks"
  - Fix useState bug
  - Debug useEffect loop
  - Resolve prop drilling
  → Changes to one affect retrieval of others
```

### 4. Adaptive Relevance

Older samples naturally decay based on coherence time:
- Recent samples: Higher retrieval probability
- Old samples: Lower probability (unless quantum tunneling used)

## Example Output

```
🔮 Initializing Q-Store quantum database...
✓ Quantum database initialized

📚 Loading training data from react_train.jsonl...
  Found 5 training samples
  Storing in quantum database with superposition contexts...
    Stored 5/5 samples

🔗 Creating entangled groups by instruction type...
  ✓ Entangled 2 samples in 'generation' group
  ✓ Entangled 1 samples in 'explanation' group
  ✓ Entangled 1 samples in 'debugging' group

✅ Loaded 5 samples into quantum database

🎯 Demonstrating quantum-enhanced data sampling:

1. Curriculum Learning (Epoch 0 - Easy samples):
   1. [easy] Create a React counter component using hooks...
   2. [easy] Build a React login form with email and password...
   3. [easy] Explain the useState hook in React...

2. Curriculum Learning (Epoch 1 - Medium samples):
   1. [medium] Create a todo list component in React...
   2. [medium] Build a React login form with email and password...
   3. [medium] Create a React counter component using hooks...

3. Context-Specific Sampling (Generation tasks):
   1. Create a React counter component using hooks...
   2. Build a React login form with email and password...
   3. Create a todo list component in React...

4. Hard Negative Mining (with quantum tunneling):
   1. Fix the infinite loop in this useEffect...
   2. Create a todo list component in React...
   3. Explain the useState hook in React...
```

## Performance Considerations

### With Quantum Backend (IonQ)

- **Superposition**: Multiple contexts per sample
- **Tunneling**: Find global patterns
- **Entanglement**: Automatic relationship management
- **Cost**: Requires IonQ API credits

### Without Quantum Backend (Classical Only)

- **Still beneficial**: Classical search with quantum-inspired algorithms
- **Entanglement**: Still works (classical correlation)
- **Curriculum**: Works with classical ranking
- **Cost**: Only Pinecone credits

## Advanced Usage

### Custom Embeddings

Replace the mock embedding function with real embeddings:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_text_embedding(text: str, dim: int = 384) -> np.ndarray:
    return model.encode(text)
```

### Dynamic Model State

Track model state during training:

```python
# Get current model embedding
def get_model_state(model):
    # Extract model representation
    # (e.g., average of layer weights)
    state = ...
    return state

# Use in sampling
model_state = get_model_state(model)
batch = await data_manager.sample_training_batch(
    batch_size=32,
    model_state=model_state,
    epoch=current_epoch
)
```

### Custom Difficulty Estimation

```python
def estimate_difficulty(sample: Dict[str, Any]) -> str:
    # Custom logic based on your data
    if 'advanced' in sample['instruction'].lower():
        return 'hard'
    # Add more sophisticated analysis
    return 'medium'
```

## Troubleshooting

### Pinecone API Key Error

```
ValueError: PINECONE_API_KEY is required
```

**Solution**: Add your Pinecone API key to `.env` file

### Transformers Not Available

```
⚠️ Transformers not installed
```

**Solution**: Install with `pip install transformers peft datasets torch` (only needed for actual training)

### Out of Memory

**Solution**: Reduce batch size or use gradient accumulation:
```python
config = TrainingConfig(
    per_device_train_batch_size=1,  # Reduce
    gradient_accumulation_steps=16  # Increase
)
```

## References

- [Q-Store Documentation](../README.md)
- [TinyLlama Model](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Quantum Computing Concepts](../quantum_db_design_v2.md)

## License

Same as Q-Store project - see [LICENSE](../LICENSE)
