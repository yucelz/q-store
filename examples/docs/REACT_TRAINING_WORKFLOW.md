# React Training Workflow with Q-Store

This guide demonstrates how to use Q-Store's quantum database for fine-tuning TinyLlama on React code generation tasks.

## 📋 Overview

The workflow consists of three main steps:
1. **Generate Dataset**: Create a comprehensive React training dataset
2. **Verify Dataset**: Check the generated dataset quality
3. **Quantum Training**: Train TinyLlama using quantum-enhanced data selection

## 🚀 Quick Start

### Step 1: Generate Dataset

```bash
cd examples
python react_dataset_generator.py
```

This generates a `react_train.jsonl` file with:
- **1,200 samples** (40%): React component generation
- **750 samples** (25%): Bug fixing examples
- **600 samples** (20%): Code explanations
- **450 samples** (15%): Code conversions
- **Total: ~3,000 samples**

### Step 2: Verify Dataset

```bash
# Check total number of samples
cat react_train.jsonl | wc -l

# View first sample
head -n 1 react_train.jsonl | python -m json.tool

# Check dataset distribution
grep -c '"instruction".*Create' react_train.jsonl  # Component generation
grep -c '"instruction".*Fix' react_train.jsonl     # Bug fixing
grep -c '"instruction".*Explain' react_train.jsonl # Explanations
grep -c '"instruction".*Convert' react_train.jsonl # Conversions
```

### Step 3: Run Quantum Training

```bash
# Make sure you have .env file with API keys
python tinyllama_react_training.py
```

## 🔮 Quantum-Enhanced Features

The training script demonstrates several quantum database capabilities:

### 1. **Curriculum Learning**
Progressively samples harder examples as training progresses:
- **Epoch 0**: Easy samples (simple components)
- **Epoch 1**: Medium samples (forms with validation)
- **Epoch 2+**: Hard samples (complex state management)

### 2. **Context-Specific Sampling**
Retrieves samples based on instruction type:
- `generation`: Component creation tasks
- `debugging`: Bug fixing examples
- `explanation`: Conceptual understanding
- `conversion`: Code refactoring

### 3. **Hard Negative Mining**
Uses quantum tunneling to find challenging examples that push model boundaries.

### 4. **Entangled Groups**
Groups related samples by instruction type for correlated retrieval.

## 📊 Dataset Structure

Each sample in `react_train.jsonl` follows this format:

```json
{
  "instruction": "Create a React counter component using hooks",
  "input": "",
  "output": "import React, { useState } from 'react';\n\nfunction Counter() {\n  const [count, setCount] = useState(0);\n  ...\n}"
}
```

## ⚙️ Configuration

Edit `TrainingConfig` in `tinyllama_react_training.py`:

```python
config = TrainingConfig(
    training_data_file="react_train.jsonl",    # Dataset file
    output_dir="./tinyllama-react-quantum",    # Model output
    max_samples=1000,                          # Samples to use
    embedding_dim=768,                         # Vector dimension
    
    # Quantum features
    use_quantum_sampling=True,
    use_curriculum_learning=True,
    use_hard_negative_mining=True,
    
    # Training hyperparameters
    num_train_epochs=3,
    per_device_train_batch_size=2,
    learning_rate=2e-4,
)
```

## 🔑 Environment Setup

Create a `.env` file in the project root:

```bash
# Required
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=us-east-1

# Optional (for quantum features)
IONQ_API_KEY=your_ionq_api_key_here
```

Get API keys:
- **Pinecone**: https://www.pinecone.io/
- **IonQ**: https://ionq.com/ (optional, for quantum simulation)

## 📦 Dependencies

Install required packages:

```bash
# Core dependencies
pip install -r requirements.txt

# For actual model training (optional)
pip install transformers peft datasets torch
```

## 🎯 Sample Output

When you run the training script, you'll see:

```
🔮 Initializing Q-Store quantum database...
✓ Quantum database initialized

📚 Loading training data from react_train.jsonl...
  Found 3000 training samples
  Storing in quantum database with superposition contexts...
  
🔗 Creating entangled groups by instruction type...
  ✓ Entangled 1200 samples in 'generation' group
  ✓ Entangled 750 samples in 'debugging' group
  
🎯 Demonstrating quantum-enhanced data sampling:

1. Curriculum Learning (Epoch 0 - Easy samples):
   1. [easy] Create a React counter component with increment and decr...
   2. [easy] Build a React search filter component for a list of item...
   3. [easy] Explain what the useState hook does in React...
```

## 📈 Customizing the Dataset

To add more samples, edit `react_dataset_generator.py`:

```python
generator = ReactDatasetGenerator()

# Adjust sample counts
generator.generate_component_samples(1500)  # More components
generator.generate_bug_fixing_samples(1000)  # More bugs
generator.generate_explanation_samples(800)  # More explanations

generator.save_to_jsonl("react_train_custom.jsonl")
```

## 🔍 Troubleshooting

### Dataset not found
```bash
# Manually generate
python react_dataset_generator.py

# Or the script will auto-generate using minimal samples
```

### API Key errors
```bash
# Check .env file exists
cat .env

# Verify keys are set
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('PINECONE_API_KEY'))"
```

### Memory issues
```python
# Reduce max_samples in config
config = TrainingConfig(
    max_samples=500,  # Use fewer samples
    per_device_train_batch_size=1,  # Smaller batches
)
```

## 📚 Next Steps

1. **Experiment with sampling strategies**: Try different query modes
2. **Add custom samples**: Extend the dataset generator
3. **Fine-tune for specific tasks**: Focus on component generation or debugging
4. **Evaluate results**: Test the fine-tuned model on new React tasks

## 🤝 Contributing

To add new React patterns or examples:

1. Add templates to `react_dataset_generator.py`
2. Regenerate dataset
3. Test with quantum training
4. Submit improvements!

---

For more information, see:
- [Q-Store Documentation](../README.md)
- [TinyLlama Training README](./TINYLLAMA_TRAINING_README.md)
- [Quantum Database Design](../quantum_db_design_v2.md)
