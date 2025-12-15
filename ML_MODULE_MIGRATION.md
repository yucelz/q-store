# Migration Guide: Core to ML Module Reorganization

## 📋 Overview

The v3.2 ML training components have been reorganized into a dedicated `ml/` module for better code organization and separation of concerns.

## 🔄 What Changed

### File Locations

| Old Location (v3.2 initial) | New Location (v3.2 final) |
|----------------------------|---------------------------|
| `src/q_store/core/quantum_layer.py` | `src/q_store/ml/quantum_layer.py` |
| `src/q_store/core/gradient_computer.py` | `src/q_store/ml/gradient_computer.py` |
| `src/q_store/core/data_encoder.py` | `src/q_store/ml/data_encoder.py` |
| `src/q_store/core/quantum_trainer.py` | `src/q_store/ml/quantum_trainer.py` |

### Import Paths

#### Before (Old)
```python
from q_store.core import (
    QuantumLayer,
    QuantumGradientComputer,
    QuantumDataEncoder,
    QuantumTrainer,
    QuantumModel,
    TrainingConfig
)
```

#### After (New) ✅
```python
from q_store.ml import (
    QuantumLayer,
    QuantumGradientComputer,
    QuantumDataEncoder,
    QuantumTrainer,
    QuantumModel,
    TrainingConfig
)
```

## 🎯 Why This Change?

### Benefits

1. **Better Organization**: ML components are now clearly separated from core database components
2. **Logical Grouping**: All training-related code is in one module
3. **Easier Maintenance**: Clear boundaries between database and ML functionality
4. **Scalability**: Room to add more ML features without cluttering core
5. **Clearer Dependencies**: Core database doesn't depend on ML components

### Module Structure

```
q_store/
├── core/          # Database operations (unchanged)
│   ├── quantum_database.py
│   ├── state_manager.py
│   ├── entanglement_registry.py
│   └── tunneling_engine.py
│
├── backends/      # Hardware abstraction (unchanged)
│   ├── quantum_backend_interface.py
│   ├── backend_manager.py
│   └── ...
│
└── ml/           # ML training (NEW location) ⭐
    ├── quantum_layer.py
    ├── gradient_computer.py
    ├── data_encoder.py
    ├── quantum_trainer.py
    └── README.md
```

## 🔧 Migration Steps

### Step 1: Update Imports

Replace all occurrences of `from q_store.core import` with `from q_store.ml import` for ML components.

#### Find and Replace

```bash
# In your code files
find . -name "*.py" -exec sed -i 's/from q_store\.core import QuantumLayer/from q_store.ml import QuantumLayer/g' {} \;
find . -name "*.py" -exec sed -i 's/from q_store\.core import QuantumTrainer/from q_store.ml import QuantumTrainer/g' {} \;
find . -name "*.py" -exec sed -i 's/from q_store\.core import QuantumGradientComputer/from q_store.ml import QuantumGradientComputer/g' {} \;
find . -name "*.py" -exec sed -i 's/from q_store\.core import QuantumDataEncoder/from q_store.ml import QuantumDataEncoder/g' {} \;
```

Or manually update each file:

```python
# Before
from q_store.core import QuantumTrainer, QuantumModel, TrainingConfig

# After
from q_store.ml import QuantumTrainer, QuantumModel, TrainingConfig
```

### Step 2: Verify Changes

Run the verification script to ensure everything works:

```bash
python verify_v3_2.py
```

Expected output:
```
✓ All ML components imported successfully
✓ QuantumLayer test passed
✓ QuantumDataEncoder test passed
✓ QuantumGradientComputer test passed
✓ QuantumTrainer test passed
✓ Examples import test passed

Passed: 6/6 tests
```

### Step 3: Update Documentation

If you have custom documentation referencing the old paths, update them:

```python
# Old documentation
"""
Import the trainer from q_store.core:
    from q_store.core import QuantumTrainer
"""

# New documentation
"""
Import the trainer from q_store.ml:
    from q_store.ml import QuantumTrainer
"""
```

## 📝 Updated Code Examples

### Basic Training

```python
import asyncio
import numpy as np
from q_store.ml import (
    QuantumTrainer,
    QuantumModel,
    TrainingConfig
)
from q_store.backends import create_default_backend_manager

async def train():
    config = TrainingConfig(
        pinecone_api_key="key",
        quantum_sdk="mock",
        learning_rate=0.01,
        epochs=10,
        n_qubits=4
    )
    
    backend_manager = create_default_backend_manager()
    trainer = QuantumTrainer(config, backend_manager)
    model = QuantumModel(4, 4, 2, backend_manager.get_backend())
    
    await trainer.train(model, data_loader)

asyncio.run(train())
```

### Using Layers Directly

```python
from q_store.ml import QuantumLayer
from q_store.backends import create_default_backend_manager

backend_manager = create_default_backend_manager()
layer = QuantumLayer(
    n_qubits=8,
    depth=4,
    backend=backend_manager.get_backend()
)

output = await layer.forward(input_data)
```

### Custom Encoding

```python
from q_store.ml import QuantumDataEncoder, QuantumFeatureMap

# Amplitude encoding
encoder = QuantumDataEncoder('amplitude')
circuit = encoder.encode(data)

# Feature map
feature_map = QuantumFeatureMap(n_qubits=8, feature_map_type='ZZFeatureMap')
circuit = feature_map.map_features(data)
```

### Gradient Computation

```python
from q_store.ml import QuantumGradientComputer

grad_computer = QuantumGradientComputer(backend)
result = await grad_computer.compute_gradients(
    circuit_builder=build_circuit,
    loss_function=compute_loss,
    parameters=params
)
```

## 🆕 What's Still Available from Core

The `q_store.core` module still provides all database-related functionality:

```python
from q_store.core import (
    QuantumDatabase,
    DatabaseConfig,
    QueryMode,
    QueryResult,
    StateManager,
    EntanglementRegistry,
    TunnelingEngine
)
```

## ⚠️ Breaking Changes

### None for Top-Level Imports

If you were importing from the top level `q_store` package, **no changes needed**:

```python
# This still works (recommended)
from q_store import (
    QuantumTrainer,
    QuantumModel,
    TrainingConfig,
    QuantumLayer
)
```

### Only Affects Direct Module Imports

Only code that specifically imported from `q_store.core` needs updating:

```python
# This no longer works ❌
from q_store.core import QuantumTrainer

# Use this instead ✅
from q_store.ml import QuantumTrainer

# Or this (also works) ✅
from q_store import QuantumTrainer
```

## 🧪 Testing Your Migration

### Quick Test

```python
# test_migration.py
try:
    # Test ML imports
    from q_store.ml import (
        QuantumTrainer,
        QuantumModel,
        QuantumLayer,
        QuantumGradientComputer,
        QuantumDataEncoder
    )
    print("✅ ML imports successful")
    
    # Test core imports (should still work)
    from q_store.core import (
        QuantumDatabase,
        DatabaseConfig
    )
    print("✅ Core imports successful")
    
    # Test top-level imports
    from q_store import QuantumTrainer as Trainer
    print("✅ Top-level imports successful")
    
    print("\n✅ All imports working correctly!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
```

### Run Comprehensive Tests

```bash
# Run verification
python verify_v3_2.py

# Run quickstart
python quickstart_v3_2.py

# Run examples
python -m q_store_examples.examples_v3_2
```

## 📚 Additional Resources

- **ML Module Documentation**: `src/q_store/ml/README.md`
- **Directory Structure**: `DIRECTORY_STRUCTURE.md`
- **Quick Start Guide**: `docs/README_v3_2.md`
- **Architecture Document**: `docs/Quantum-Native_Database_Architecture_v3_2.md`

## 💡 Tips

1. **Use IDE Find/Replace**: Most IDEs can find and replace across multiple files
2. **Test Incrementally**: Update and test one module at a time
3. **Use Top-Level Imports**: Import from `q_store` directly to future-proof your code
4. **Check Examples**: Look at `examples_v3_2.py` for reference

## 🆘 Troubleshooting

### Issue: "No module named 'q_store.core.quantum_trainer'"

**Solution**: Update import to `from q_store.ml import QuantumTrainer`

### Issue: "Cannot import name 'QuantumLayer' from 'q_store.core'"

**Solution**: Change to `from q_store.ml import QuantumLayer`

### Issue: All imports fail

**Solution**: Reinstall the package:
```bash
pip install -e .
```

## ✅ Migration Checklist

- [ ] Update all ML component imports from `q_store.core` to `q_store.ml`
- [ ] Run `python verify_v3_2.py` to verify imports
- [ ] Run your test suite
- [ ] Run `python quickstart_v3_2.py` for end-to-end test
- [ ] Update documentation/comments referencing old paths
- [ ] Verify production code still works
- [ ] Update CI/CD pipelines if needed

## 📊 Summary

| Aspect | Old | New |
|--------|-----|-----|
| ML Components Location | `q_store.core.*` | `q_store.ml.*` |
| Import Statement | `from q_store.core import ...` | `from q_store.ml import ...` |
| Files Moved | 4 files | To `ml/` module |
| Breaking Changes | Minimal | Only direct imports |
| Top-Level Imports | ✅ Still work | ✅ Still work |

---

**Migration Status**: ✅ Complete  
**Backward Compatibility**: ✅ Top-level imports unchanged  
**Recommended Action**: Update direct module imports  
**Urgency**: Low (top-level imports still work)
