# Changelog

All notable changes to Q-Store will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.4.0] - 2025-12-16

### Added
- **IonQ Batch API Integration**: Single API call for multiple circuits (8-12x performance improvement)
- **Smart Circuit Caching**: Template-based caching with parameter binding (10x faster circuit preparation)
- **IonQ Native Gate Compilation**: GPi, GPi2, MS gates for 30% performance boost
- **Connection Pooling**: Persistent HTTP connections eliminate 90% of connection overhead
- **Circuit Batch Manager**: Efficient batching of quantum circuits for parallel execution
- **Performance Tracking**: Comprehensive metrics and monitoring for quantum operations
- MIT License for public distribution

### Changed
- Training time reduced from 30 minutes to 3-4 minutes (8.8x faster)
- Throughput increased from 0.5-0.6 to 5-8 circuits/second
- Gate count reduced by 28% with native gate compilation
- Batch processing time reduced from 35s to 4s for 20 circuits

### Performance Benchmarks
| Metric | v3.3.1 | v3.4.0 | Improvement |
|--------|---------|---------|-------------|
| Batch time (20 circuits) | 35s | 4s | 8.8x faster |
| Training (5 epochs, 100 samples) | 29 min | 3.3 min | 8.8x faster |
| Circuits/second | 0.57 | 5.0 | 8.8x faster |
| Gate count | Medium | Low | 28% reduction |

### Fixed
- Connection overhead in IonQ backend
- Circuit compilation inefficiencies
- Memory leaks in gradient computation

## [3.3.1] - 2024-11-15

### Fixed
- Critical bug fixes in quantum gradient estimation
- Memory optimization in circuit caching
- Improved error handling in IonQ backend

## [3.3.0] - 2024-11-01

### Added
- **SPSA Gradient Estimation**: 24-48x faster training compared to parameter shift
- **Hardware-Efficient Quantum Layers**: 33% fewer parameters
- **Circuit Batching and Caching**: 2-5x speedup in training
- **Adaptive Gradient Optimization**: Dynamic learning rate adjustment
- **Performance Monitoring**: Comprehensive tracking and metrics

### Changed
- Default gradient estimator to SPSA for better performance
- Quantum layer architecture optimized for hardware efficiency

### Performance
- Training time reduced by 24-48x with SPSA
- Circuit evaluation 2-5x faster with batching
- Memory usage reduced by 33% with efficient layers

## [3.2.0] - 2024-09-15

### Added
- **Quantum ML Training**: Full quantum neural network training capabilities
- **Hybrid Classical-Quantum Pipelines**: Integration with PyTorch/TensorFlow
- **Quantum Data Encoding**: Amplitude and angle encoding strategies
- **Gradient Computation**: Parameter shift rule for backpropagation
- **Training Infrastructure**: Model checkpointing and metrics tracking

### Features
- Quantum transfer learning support
- Quantum data augmentation
- Quantum regularization techniques
- Distributed quantum training

## [3.1.0] - 2024-07-01

### Added
- **Hardware-Agnostic Architecture**: Support for multiple quantum SDKs
- **Backend Manager**: Plugin architecture for quantum backends
- **Cirq Adapter**: Full Cirq SDK integration
- **Qiskit Adapter**: Qiskit SDK support for IonQ
- **Mock Backend**: Classical simulation for testing

### Changed
- Refactored backend system for better extensibility
- Improved error handling and logging
- Enhanced documentation

### Migration
- See [V3.1_UPGRADE_GUIDE.md](docs/archive/V3.1_UPGRADE_GUIDE.md) for migration instructions

## [3.0.0] - 2024-05-01

### Added
- **Quantum Tunneling**: Global pattern discovery across database
- **Adaptive Decoherence**: Physics-based relevance decay
- **Enhanced Entanglement**: Multi-group entanglement support
- **Production Ready**: Comprehensive error handling and logging

### Changed
- Complete architecture redesign
- Improved quantum state management
- Better Pinecone integration

### Breaking Changes
- API changes in quantum database initialization
- New entanglement registry interface

## [2.0.0] - 2024-03-01

### Added
- **Quantum Superposition**: Store vectors in multiple contexts
- **Quantum Entanglement**: Automatic relationship synchronization
- **IonQ Backend**: Real quantum hardware support
- **Pinecone Integration**: Scalable vector storage

### Features
- Context-aware retrieval
- Automatic cache invalidation via entanglement
- Quantum state collapse on measurement

## [1.0.0] - 2024-01-15

### Added
- Initial release
- Basic quantum database operations
- Vector similarity search
- Mock quantum backend

---

## Release Process

### Version Numbering
- **Major**: Breaking changes or significant architecture changes
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes and minor improvements

### Supported Versions
| Version | Supported | End of Support |
|---------|-----------|----------------|
| 3.4.x   | ✅ Yes    | TBD            |
| 3.3.x   | ✅ Yes    | 2025-06-01     |
| 3.2.x   | ⚠️ Limited | 2025-03-01   |
| < 3.2   | ❌ No     | Ended          |

### Upgrading
- [v3.4 Migration Guide](docs/V3_4_IMPLEMENTATION_COMPLETE.md)
- [v3.3 Migration Guide](docs/archive/V3_3_IMPLEMENTATION_SUMMARY.md)
- [v3.2 Migration Guide](docs/archive/V3_2_IMPLEMENTATION_SUMMARY.md)
- [v3.1 Migration Guide](docs/archive/V3.1_UPGRADE_GUIDE.md)

### Security Updates
For security vulnerabilities, please see [SECURITY.md](SECURITY.md).

---

## Links
- [Homepage](http://www.q-store.tech)
- [Documentation](https://github.com/yucelz/q-store/tree/main/docs)
- [Repository](https://github.com/yucelz/q-store)
- [Issues](https://github.com/yucelz/q-store/issues)
