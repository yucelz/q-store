# Q-Store Examples

Standalone example projects demonstrating Q-Store quantum database capabilities for ML training, financial applications, and more.

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- API keys (see [API Keys](#api-keys) section)

### Installation

#### Option 1: Using pip (Recommended)

```bash
# 1. Clone the repository (if not already done)
git clone https://github.com/yucelz/q-store.git
cd q-store

# 2. Install Q-Store core package
pip install -e .

# 3. Navigate to examples directory
cd examples

# 4. Install example dependencies
pip install -r requirements.txt

# 5. Set up environment variables
cp .env.example .env
# Edit .env and add your API keys

# 6. Verify installation
python verify_installation.py
```

#### Option 2: Using conda

```bash
# 1. Clone the repository
git clone https://github.com/yucelz/q-store.git
cd q-store/examples

# 2. Create conda environment
conda env create -f environment.yml

# 3. Activate environment
conda activate q-store-examples

# 4. Install Q-Store from parent directory
cd ..
pip install -e .
cd examples

# 5. Set up environment variables
cp .env.example .env
# Edit .env and add your API keys

# 6. Verify installation
python verify_installation.py
```

#### Option 3: Minimal Installation (No ML Dependencies)

```bash
# Install only core dependencies
pip install -r requirements-minimal.txt

# This allows running:
# - basic_example.py
# - financial_example.py
# - quantum_db_quickstart.py
```

## 🔑 API Keys

### Required

- **Pinecone API Key**: Required for all examples
  - Get it from: https://www.pinecone.io/
  - Free tier available (100K vectors)

### Optional

- **IonQ API Key**: Optional, enables quantum simulation features
  - Get it from: https://ionq.com/
  - Free credits available for new users

- **Hugging Face Token**: Optional, only for gated models
  - Get it from: https://huggingface.co/settings/tokens

### Configuration

```bash
# Copy the example file
cp .env.example .env

# Edit with your favorite editor
nano .env  # or vim, code, etc.

# Add your keys:
PINECONE_API_KEY=your_actual_key_here
PINECONE_ENVIRONMENT=us-east-1
IONQ_API_KEY=your_ionq_key_here  # Optional
```

## 📚 Available Examples

### 1. Basic Example (`basic_example.py`)

Demonstrates core Q-Store functionality:
- Inserting vectors with quantum contexts
- Querying with superposition
- Creating entangled groups
- Quantum tunneling for exploration

```bash
python basic_example.py
```

### 2. Financial Example (`financial_example.py`)

Financial data analysis with quantum features:
- Portfolio optimization
- Risk correlation analysis
- Market regime detection
- Anomaly detection

```bash
python financial_example.py
```

### 3. Quantum Database Quickstart (`quantum_db_quickstart.py`)

Comprehensive tutorial covering:
- Database initialization
- All query modes (PRECISE, BALANCED, EXPLORATORY)
- Advanced quantum features
- Performance optimization

```bash
python quantum_db_quickstart.py
```

### 4. ML Training Example (`ml_training_example.py`)

Machine learning integration:
- Model embedding storage
- Training data selection
- Curriculum learning
- Hard negative mining

```bash
python ml_training_example.py
```

### 5. TinyLlama React Training (`tinyllama_react_training.py`)

Complete LLM fine-tuning workflow:
- React code dataset generation
- Quantum-enhanced data sampling
- LoRA fine-tuning
- Curriculum learning

```bash
# Option 1: Automated workflow
./run_react_training.sh

# Option 2: Step-by-step
python react_dataset_generator.py
python tinyllama_react_training.py

# See REACT_QUICK_REFERENCE.md for details
```

## 📖 Documentation

| Document | Description |
|----------|-------------|
| **REACT_QUICK_REFERENCE.md** | Quick start for React training |
| **REACT_TRAINING_WORKFLOW.md** | Detailed React training guide |
| **TINYLLAMA_TRAINING_README.md** | TinyLlama fine-tuning guide |
| **IMPROVEMENTS_SUMMARY.md** | Code improvements and comparisons |

## 🛠️ Project Structure

```
examples/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── requirements-minimal.txt       # Minimal dependencies
├── environment.yml                # Conda environment
├── pyproject.toml                 # Modern Python packaging
├── setup.py                       # Package setup
├── .env.example                   # Environment template
│
├── basic_example.py               # Basic Q-Store usage
├── financial_example.py           # Financial applications
├── quantum_db_quickstart.py       # Comprehensive tutorial
├── ml_training_example.py         # ML integration
│
├── tinyllama_react_training.py    # React LLM training
├── react_dataset_generator.py     # Dataset generation
├── run_react_training.sh          # Automation script
├── verify_react_integration.py    # Integration tests
├── verify_tinyllama_example.py    # TinyLlama verification
│
└── [Documentation files...]
```

## 🔧 Configuration

### Environment Variables

All configuration is done through `.env` file:

```bash
# Required
PINECONE_API_KEY=your_key
PINECONE_ENVIRONMENT=us-east-1

# Optional
IONQ_API_KEY=your_ionq_key
IONQ_TARGET=simulator

# ML Training (optional)
HUGGING_FACE_TOKEN=your_token
OUTPUT_DIR=./models
```

### Custom Settings

Edit configuration in each example file:

```python
# Example: tinyllama_react_training.py
config = TrainingConfig(
    max_samples=1000,
    num_train_epochs=3,
    use_quantum_sampling=True,
    use_curriculum_learning=True
)
```

## 🧪 Testing

### Verify Installation

```bash
# Test Q-Store installation
python verify_installation.py

# Test React integration
python verify_react_integration.py

# Test TinyLlama setup
python verify_tinyllama_example.py
```

### Run Unit Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# With coverage
pytest --cov=. --cov-report=html
```

## 💡 Usage Tips

### GPU Support

For CUDA GPU support:

```bash
# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU availability
python -c "import torch; print('GPU Available:', torch.cuda.is_available())"
```

### Memory Management

For large datasets or limited RAM:

```python
# Reduce batch size
config = TrainingConfig(
    per_device_train_batch_size=1,  # Smaller batches
    gradient_accumulation_steps=16,  # Accumulate gradients
    max_samples=500                   # Limit dataset size
)
```

### Development Mode

Install in development mode for easy editing:

```bash
# Install Q-Store in editable mode
cd ..
pip install -e .

# Now changes to q_store source are immediately available
```

## 🐛 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: q_store` | Install Q-Store: `cd .. && pip install -e .` |
| `PINECONE_API_KEY not found` | Create `.env` file with your API key |
| `ImportError: transformers` | Install ML dependencies: `pip install -r requirements.txt` |
| `CUDA out of memory` | Reduce batch size or use CPU |
| `Dataset file not found` | Run dataset generator first |

### Debug Mode

Enable verbose logging:

```bash
# Set environment variable
export LOG_LEVEL=DEBUG

# Or in .env file
echo "LOG_LEVEL=DEBUG" >> .env
```

### Getting Help

1. Check the documentation files in this directory
2. Review the parent [Q-Store README](../README.md)
3. Open an issue on GitHub
4. Review existing issues and discussions

## 🎯 Next Steps

1. **Run Basic Example**: Start with `basic_example.py`
2. **Try React Training**: Use the automated workflow
3. **Experiment**: Modify configs and try different strategies
4. **Build Your Own**: Use examples as templates
5. **Contribute**: Share improvements and new examples

## 📊 Performance Benchmarks

### Dataset Sizes

- **Minimal**: 500-1,000 samples (fast, for testing)
- **Medium**: 1,000-5,000 samples (balanced)
- **Large**: 5,000-10,000+ samples (best results)

### Training Times (approximate)

- **Dataset Generation**: 10-30 seconds
- **Database Loading**: 1-3 minutes
- **Quantum Sampling Demo**: 30 seconds
- **Full Training**: 30-90 minutes (with GPU)

## 🤝 Contributing

Contributions are welcome! To add new examples:

1. Follow the existing code structure
2. Add documentation
3. Include requirements
4. Test thoroughly
5. Submit a pull request

## 📄 License

MIT License - see parent repository for details

## 🔗 Related Resources

- [Q-Store Main Repository](https://github.com/yucelz/q-store)
- [Quantum Database Design](../quantum_db_design_v2.md)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [IonQ Documentation](https://ionq.com/docs/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)

## 📞 Support

- GitHub Issues: [q-store/issues](https://github.com/yucelz/q-store/issues)
- Documentation: [examples/](https://github.com/yucelz/q-store/tree/main/examples)

---

**Ready to start?** Run the verification script:

```bash
python verify_installation.py
```

If all checks pass, you're ready to explore quantum-enhanced machine learning! 🚀
