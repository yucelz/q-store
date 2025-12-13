# Quick Reference: React Training with Q-Store

## 🚀 Three Ways to Run

### Option 1: Automated Script (Recommended)
```bash
cd examples
./run_react_training.sh
```

### Option 2: Step-by-Step Manual
```bash
cd examples

# Step 1: Generate dataset
python react_dataset_generator.py

# Step 2: Verify dataset
cat react_train.jsonl | wc -l  # Should show 3000+

# Step 3: Run training
python tinyllama_react_training.py
```

### Option 3: Individual Steps
```bash
# Only generate dataset
python react_dataset_generator.py

# Only run training (auto-generates dataset if missing)
python tinyllama_react_training.py
```

## 📁 Files Overview

| File | Purpose |
|------|---------|
| `react_dataset_generator.py` | Generates 3000+ React training samples |
| `tinyllama_react_training.py` | Quantum-enhanced training script |
| `run_react_training.sh` | Automated workflow script |
| `REACT_TRAINING_WORKFLOW.md` | Detailed documentation |

## 🔑 Required Setup

### 1. Environment Variables (.env)
```bash
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=us-east-1
IONQ_API_KEY=your_ionq_key  # Optional
```

### 2. Dependencies
```bash
# Core
pip install -r requirements.txt

# For actual training (optional)
pip install transformers peft datasets torch
```

## 📊 What You Get

### Dataset (`react_train.jsonl`)
- **~3,000 samples** in JSONL format
- 40% component generation
- 25% bug fixing
- 20% explanations
- 15% conversions

### Quantum Features
1. **Curriculum Learning**: Easy → Hard progression
2. **Context-Specific Sampling**: By task type
3. **Hard Negative Mining**: Challenging examples
4. **Entangled Groups**: Related samples

### Output
- Fine-tuned model: `./tinyllama-react-quantum/`
- Training logs and checkpoints
- Quantum sampling demonstrations

## 🎯 Quick Test

```bash
# Test dataset generation
python -c "
from react_dataset_generator import ReactDatasetGenerator
gen = ReactDatasetGenerator()
gen.generate_all()
print(f'Generated {len(gen.samples)} samples')
"

# Test quantum database
python -c "
import asyncio
from tinyllama_react_training import QuantumTrainingDataManager, TrainingConfig

async def test():
    config = TrainingConfig(max_samples=10)
    manager = QuantumTrainingDataManager(config)
    await manager.initialize()
    print('✓ Quantum database initialized')
    await manager.close()

asyncio.run(test())
"
```

## 🐛 Common Issues

| Issue | Solution |
|-------|----------|
| `react_train.jsonl not found` | Run `python react_dataset_generator.py` |
| `PINECONE_API_KEY not set` | Create `.env` file with API key |
| `Transformers not available` | Script runs demo without training |
| `Memory error` | Reduce `max_samples` in config |

## 📈 Performance Tips

1. **More Data**: Increase sample counts in generator
2. **Better Sampling**: Enable all quantum features
3. **Curriculum Learning**: Let model learn progressively
4. **GPU Training**: Use CUDA if available

## 🔗 Related Docs

- [Full Workflow Guide](./REACT_TRAINING_WORKFLOW.md)
- [TinyLlama Training](./TINYLLAMA_TRAINING_README.md)
- [Q-Store Documentation](../README.md)
- [Quantum Database Design](../quantum_db_design_v2.md)

---

**Need Help?** Check the detailed guide: [REACT_TRAINING_WORKFLOW.md](./REACT_TRAINING_WORKFLOW.md)
