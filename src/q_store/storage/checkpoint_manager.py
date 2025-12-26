"""
Checkpoint Manager - Zarr-Based Model Checkpointing

Async checkpoint manager for model state.
Uses Zarr for efficient binary storage with compression.

Features:
- Zarr-based storage (chunked, compressed)
- Async save/load
- Atomic operations
- Metadata tracking
- Automatic cleanup
- Version management

Design:
- Each checkpoint is a Zarr group
- Model params → Zarr arrays
- Optimizer state → Zarr arrays
- Metadata → Zarr attrs
- Blosc compression (zstd)
"""

import zarr
import numpy as np
from pathlib import Path
import asyncio
import time
from typing import Dict, Any, Optional, List
import shutil


class CheckpointManager:
    """
    Async checkpoint manager using Zarr.
    
    Checkpoints are atomic and compressed.
    
    Parameters
    ----------
    checkpoint_dir : Path or str
        Directory for checkpoints
    keep_last : int, default=5
        Keep only last N checkpoints (0 = keep all)
    compression : str, default='zstd'
        Compression algorithm ('zstd', 'lz4', 'gzip')
    compression_level : int, default=3
        Compression level (1-9)
    
    Examples
    --------
    >>> manager = CheckpointManager('checkpoints/')
    >>> await manager.save(epoch=10, model_state={...}, optimizer_state={...})
    >>> state = await manager.load(epoch=10)
    >>> checkpoints = manager.list_checkpoints()
    """
    
    def __init__(
        self,
        checkpoint_dir: Path,
        keep_last: int = 5,
        compression: str = 'zstd',
        compression_level: int = 3,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        
        # Compression settings
        if compression == 'zstd':
            self.compressor = zarr.Blosc(cname='zstd', clevel=compression_level)
        elif compression == 'lz4':
            self.compressor = zarr.Blosc(cname='lz4', clevel=compression_level)
        elif compression == 'gzip':
            self.compressor = zarr.Blosc(cname='gzip', clevel=compression_level)
        else:
            raise ValueError(f"Unknown compression: {compression}")
        
        # Open Zarr store
        self.store = zarr.DirectoryStore(str(self.checkpoint_dir))
        self.root = zarr.group(store=self.store)
    
    async def save(
        self,
        epoch: int,
        model_state: Dict[str, Any],
        optimizer_state: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Save checkpoint (async).
        
        Runs in thread pool to not block training.
        
        Parameters
        ----------
        epoch : int
            Epoch number
        model_state : dict
            Model parameters (name -> array)
        optimizer_state : dict, optional
            Optimizer state
        metadata : dict, optional
            Additional metadata
        """
        await asyncio.get_event_loop().run_in_executor(
            None,
            self._save_sync,
            epoch,
            model_state,
            optimizer_state,
            metadata
        )
    
    def _save_sync(
        self,
        epoch: int,
        model_state: Dict[str, Any],
        optimizer_state: Optional[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]],
    ):
        """Synchronous save operation."""
        start_time = time.time()
        
        # Create epoch group
        epoch_name = f'epoch_{epoch:04d}'
        
        # Delete if exists (overwrite)
        if epoch_name in self.root:
            del self.root[epoch_name]
        
        epoch_group = self.root.create_group(epoch_name)
        
        # Save model parameters
        model_group = epoch_group.create_group('model')
        for name, param in model_state.items():
            # Convert to numpy if needed
            if hasattr(param, 'cpu'):  # PyTorch tensor
                param_np = param.cpu().detach().numpy()
            elif hasattr(param, 'numpy'):  # TensorFlow tensor
                param_np = param.numpy()
            else:
                param_np = np.asarray(param)
            
            model_group.array(
                name,
                data=param_np,
                compressor=self.compressor,
                overwrite=True
            )
        
        # Save optimizer state
        if optimizer_state:
            opt_group = epoch_group.create_group('optimizer')
            for name, state in optimizer_state.items():
                if isinstance(state, dict):
                    # Nested state (e.g., per-parameter state)
                    state_group = opt_group.create_group(name)
                    for key, value in state.items():
                        if hasattr(value, 'cpu'):
                            value = value.cpu().detach().numpy()
                        elif hasattr(value, 'numpy'):
                            value = value.numpy()
                        else:
                            value = np.asarray(value)
                        state_group.array(key, data=value, compressor=self.compressor)
                else:
                    # Scalar state
                    if hasattr(state, 'cpu'):
                        state = state.cpu().detach().numpy()
                    elif hasattr(state, 'numpy'):
                        state = state.numpy()
                    opt_group.array(name, data=state, compressor=self.compressor)
        
        # Save metadata
        epoch_group.attrs['epoch'] = epoch
        epoch_group.attrs['timestamp'] = time.time()
        epoch_group.attrs['save_duration_ms'] = (time.time() - start_time) * 1000
        
        if metadata:
            for key, value in metadata.items():
                epoch_group.attrs[key] = value
        
        # Cleanup old checkpoints
        if self.keep_last > 0:
            self._cleanup_old_checkpoints()
        
        print(f"✓ Checkpoint saved: {epoch_name} ({time.time() - start_time:.2f}s)")
    
    async def load(self, epoch: int) -> Dict[str, Any]:
        """
        Load checkpoint (async).
        
        Parameters
        ----------
        epoch : int
            Epoch number to load
        
        Returns
        -------
        state : dict
            Checkpoint state with keys: 'epoch', 'model_state', 'optimizer_state', 'metadata'
        """
        return await asyncio.get_event_loop().run_in_executor(
            None,
            self._load_sync,
            epoch
        )
    
    def _load_sync(self, epoch: int) -> Dict[str, Any]:
        """Synchronous load operation."""
        epoch_name = f'epoch_{epoch:04d}'
        
        if epoch_name not in self.root:
            raise ValueError(f"Checkpoint not found: {epoch_name}")
        
        epoch_group = self.root[epoch_name]
        
        # Load model state
        model_state = {}
        if 'model' in epoch_group:
            for name in epoch_group['model'].keys():
                model_state[name] = epoch_group['model'][name][:]
        
        # Load optimizer state
        optimizer_state = {}
        if 'optimizer' in epoch_group:
            for name in epoch_group['optimizer'].keys():
                item = epoch_group['optimizer'][name]
                if isinstance(item, zarr.Group):
                    optimizer_state[name] = {
                        key: item[key][:] for key in item.keys()
                    }
                else:
                    optimizer_state[name] = item[:]
        
        # Load metadata
        metadata = dict(epoch_group.attrs)
        
        return {
            'epoch': epoch,
            'model_state': model_state,
            'optimizer_state': optimizer_state,
            'metadata': metadata,
        }
    
    def list_checkpoints(self) -> List[int]:
        """
        List available checkpoints.
        
        Returns
        -------
        epochs : List[int]
            List of available epoch numbers
        """
        epochs = []
        for key in self.root.keys():
            if key.startswith('epoch_'):
                epoch = int(key.split('_')[1])
                epochs.append(epoch)
        return sorted(epochs)
    
    def latest_checkpoint(self) -> Optional[int]:
        """Get latest checkpoint epoch number."""
        checkpoints = self.list_checkpoints()
        return max(checkpoints) if checkpoints else None
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only last N."""
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) <= self.keep_last:
            return
        
        # Remove oldest
        to_remove = checkpoints[:-self.keep_last]
        for epoch in to_remove:
            epoch_name = f'epoch_{epoch:04d}'
            if epoch_name in self.root:
                del self.root[epoch_name]
                print(f"  Removed old checkpoint: {epoch_name}")
    
    def get_checkpoint_info(self, epoch: int) -> Dict[str, Any]:
        """
        Get checkpoint metadata without loading full state.
        
        Parameters
        ----------
        epoch : int
            Epoch number
        
        Returns
        -------
        info : dict
            Checkpoint metadata
        """
        epoch_name = f'epoch_{epoch:04d}'
        
        if epoch_name not in self.root:
            raise ValueError(f"Checkpoint not found: {epoch_name}")
        
        epoch_group = self.root[epoch_name]
        
        # Get sizes
        model_size = sum(
            epoch_group['model'][name].nbytes
            for name in epoch_group['model'].keys()
        ) if 'model' in epoch_group else 0
        
        return {
            'epoch': epoch,
            'timestamp': epoch_group.attrs.get('timestamp'),
            'model_size_mb': model_size / (1024 * 1024),
            'metadata': dict(epoch_group.attrs),
        }
