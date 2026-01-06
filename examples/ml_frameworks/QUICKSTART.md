# Image Classification from Scratch - Quick Start Guide

## 🚀 5-Minute Quick Start

Get started with quantum-enhanced image classification in just 5 minutes!

### Step 1: Check Dependencies (30 seconds)

```bash
cd /path/to/q-store
python examples/ml_frameworks/check_image_classification_deps.py
```

**Expected output:**
```
✓ All required dependencies are available!
🎉 You can run the image classification example
```

### Step 2: Run Quick Test (5 minutes)

```bash
python examples/ml_frameworks/image_classification_from_scratch.py --quick-test --visualize
```

**What happens:**
1. Downloads Cats vs Dogs dataset (~800 MB, first time only)
2. Trains on 1,000 samples for 5 epochs
3. Uses quantum feature extractor with 8 qubits
4. Generates visualization plots
5. Saves trained model

**Expected results:**
- Training time: ~5-10 minutes
- Validation accuracy: 75-85%
- Generated files:
  - `examples/ml_frameworks/training_samples.png`
  - `examples/ml_frameworks/augmented_samples.png`
  - `examples/ml_frameworks/training_history.png`
  - `examples/ml_frameworks/sample_predictions.png`

### Step 3: View Results

```bash
# Open generated images
ls -lh examples/ml_frameworks/*.png

# View training log
cat checkpoints/cats_vs_dogs/training_log.csv
```

## 📊 Command Reference

### Basic Commands

```bash
# Quick test with visualization
python examples/ml_frameworks/image_classification_from_scratch.py --quick-test --visualize

# Full training (1-2 hours)
python examples/ml_frameworks/image_classification_from_scratch.py --visualize

# Classical-only baseline
python examples/ml_frameworks/image_classification_from_scratch.py --no-quantum --quick-test

# Download dataset only
python examples/ml_frameworks/image_classification_from_scratch.py --download-only
```

### Advanced Options

```bash
# Custom epochs and batch size
python examples/ml_frameworks/image_classification_from_scratch.py \
    --epochs 50 \
    --batch-size 64 \
    --visualize

# Real quantum hardware (requires IONQ_API_KEY)
python examples/ml_frameworks/image_classification_from_scratch.py \
    --no-mock \
    --quick-test
```

## 🎯 What You'll Learn

### 1. Quantum-Classical Hybrid ML
- How to integrate quantum layers into classical CNNs
- When to use quantum vs classical layers
- Performance trade-offs

### 2. Data Preprocessing
- Image loading and augmentation
- Dataset splitting and validation
- Efficient data pipelines

### 3. Model Architecture
- Residual connections
- Quantum feature extraction
- Multi-basis measurements

### 4. Training Best Practices
- Early stopping
- Learning rate scheduling
- Model checkpointing

## 📁 File Structure

```
examples/ml_frameworks/
├── image_classification_from_scratch.py     # Main example (900+ lines)
├── IMAGE_CLASSIFICATION_README.md           # Detailed documentation
├── KERAS_VS_QSTORE_COMPARISON.md           # Comparison with Keras
├── check_image_classification_deps.py       # Dependency checker
└── QUICKSTART.md                            # This file

Generated outputs:
├── training_samples.png                     # Training data samples
├── augmented_samples.png                    # Augmented samples
├── training_history.png                     # Training curves
└── sample_predictions.png                   # Model predictions

checkpoints/cats_vs_dogs/
├── model_epoch_*.keras                      # Epoch checkpoints
├── final_model.keras                        # Final trained model
└── training_log.csv                         # Training metrics
```

## 🔍 Understanding the Output

### Training Output

```
Epoch 1/5
 625/625 [==============================] - 45s 72ms/step - loss: 0.6523 - acc: 0.6125 - val_loss: 0.5834 - val_acc: 0.6850
Epoch 2/5
 625/625 [==============================] - 43s 69ms/step - loss: 0.5234 - acc: 0.7350 - val_loss: 0.4923 - val_acc: 0.7625
...
```

**What it means:**
- `loss`: Training loss (lower is better)
- `acc`: Training accuracy (higher is better)
- `val_loss`: Validation loss
- `val_acc`: Validation accuracy (most important metric)

### Visualization Files

1. **training_samples.png**: Shows 9 random training images
2. **augmented_samples.png**: Shows data augmentation effects
3. **training_history.png**: Plots loss and accuracy curves
4. **sample_predictions.png**: Shows model predictions on test images

## 🎨 Example Results

### Quick Test Mode
```
Training time: ~5-10 minutes
Final validation accuracy: 75-85%
Best validation accuracy: ~80%
Model size: ~50 MB
```

### Full Training Mode
```
Training time: ~1-2 hours
Final validation accuracy: 90-93%
Best validation accuracy: ~92%
Model size: ~50 MB
```

## 🔧 Troubleshooting

### Problem: Import errors

**Solution:**
```bash
pip install tensorflow>=2.13.0 numpy matplotlib python-dotenv
cd /path/to/q-store
pip install -e .
```

### Problem: Out of memory

**Solution:**
```bash
python examples/ml_frameworks/image_classification_from_scratch.py \
    --batch-size 16 \
    --quick-test
```

### Problem: Download fails

**Solution:**
```bash
# Download manually
wget https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip
mkdir -p data/cats_vs_dogs
unzip kagglecatsanddogs_5340.zip -d data/cats_vs_dogs/
```

### Problem: Slow training

**Solution:**
```bash
# Use classical-only mode
python examples/ml_frameworks/image_classification_from_scratch.py \
    --no-quantum \
    --quick-test
```

## 📚 Next Steps

### 1. Explore the Code
```bash
# Open the main example
code examples/ml_frameworks/image_classification_from_scratch.py

# Read the detailed documentation
code examples/ml_frameworks/IMAGE_CLASSIFICATION_README.md
```

### 2. Try Variations

**Adjust quantum parameters:**
```python
Config.n_qubits = 10  # Use more qubits
Config.quantum_depth = 5  # Deeper quantum circuits
```

**Change architecture:**
```python
# In make_quantum_model(), add more quantum layers
x = QuantumFeatureExtractor(n_qubits=8, depth=3)(x)
x = QuantumPooling(n_qubits=8, pool_size=2)(x)
x = QuantumFeatureExtractor(n_qubits=4, depth=2)(x)
```

### 3. Compare Performance

```bash
# Run quantum version
python examples/ml_frameworks/image_classification_from_scratch.py --quick-test > quantum_log.txt

# Run classical version
python examples/ml_frameworks/image_classification_from_scratch.py --no-quantum --quick-test > classical_log.txt

# Compare accuracy
grep "val_acc" quantum_log.txt
grep "val_acc" classical_log.txt
```

### 4. Use Real Quantum Hardware

```bash
# Set up API key
echo "IONQ_API_KEY=your_key_here" >> examples/.env

# Run on real hardware
python examples/ml_frameworks/image_classification_from_scratch.py \
    --no-mock \
    --quick-test
```

⚠️ **Warning:** This consumes API credits!

## 💡 Tips for Success

### 1. Start Small
- Use `--quick-test` first
- Verify everything works
- Then scale up to full training

### 2. Monitor Resources
```bash
# Check GPU usage (if using GPU)
nvidia-smi

# Check memory usage
top

# Check disk space
df -h
```

### 3. Save Experiments
```bash
# Create experiment directory
mkdir -p experiments/my_experiment

# Copy training log
cp checkpoints/cats_vs_dogs/training_log.csv experiments/my_experiment/

# Save configuration
python -c "from examples.ml_frameworks.image_classification_from_scratch import Config; print(vars(Config))" > experiments/my_experiment/config.txt
```

### 4. Iterate Quickly
```bash
# Quick iteration loop
while true; do
    # Modify hyperparameters in code
    
    # Run quick test
    python examples/ml_frameworks/image_classification_from_scratch.py --quick-test
    
    # Check results
    tail checkpoints/cats_vs_dogs/training_log.csv
    
    # Continue if promising
    read -p "Continue with full training? (y/n) " -n 1 -r
    [[ $REPLY =~ ^[Yy]$ ]] && break
done
```

## 🎓 Learning Resources

### Documentation
- [Detailed README](IMAGE_CLASSIFICATION_README.md)
- [Keras Comparison](KERAS_VS_QSTORE_COMPARISON.md)
- [Q-Store Architecture](../../docs/Q-STORE_V4_1_1_ARCHITECTURE_DESIGN.md)

### Related Examples
- [Fashion MNIST Plain](fashion_mnist_plain.py)
- [TensorFlow Integration](tensorflow/fashion_mnist_tensorflow.py)
- [PyTorch Integration](pytorch/fashion_mnist_pytorch.py)

### External Resources
- [Original Keras Example](https://keras.io/examples/vision/image_classification_from_scratch/)
- [Keras Vision Examples](https://keras.io/examples/vision/)
- [Quantum Machine Learning](https://pennylane.ai/qml/)

## ✅ Success Checklist

- [ ] Dependencies installed and checked
- [ ] Dataset downloaded (~800 MB)
- [ ] Quick test runs successfully
- [ ] Visualization files generated
- [ ] Training log created
- [ ] Model saved to checkpoints/
- [ ] Validation accuracy > 75%

## 🎉 Congratulations!

You've successfully run quantum-enhanced image classification!

**What you achieved:**
- ✅ Built a quantum-classical hybrid CNN
- ✅ Trained on real image data
- ✅ Used quantum feature extraction
- ✅ Achieved competitive accuracy

**Next challenges:**
- 🚀 Run full training (90%+ accuracy)
- 🔬 Experiment with quantum parameters
- 🏆 Compare quantum vs classical performance
- 🌐 Deploy on real quantum hardware

---

**Questions?** Check the [detailed README](IMAGE_CLASSIFICATION_README.md) or open an issue on GitHub!
