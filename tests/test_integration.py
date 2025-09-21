#!/usr/bin/env python3
"""
Integration tests for distributed training framework
Tests end-to-end workflows and multi-component interactions
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import tempfile
import shutil
from pathlib import Path

from src.core.distributed_training import SimpleResNet
from src.core.enhanced_trainer import EnhancedDistributedTrainer
from src.monitoring.monitoring_dashboard import DistributedMonitor
from src.config.config_manager import ConfigManager, ExperimentConfig
from src.utils import set_seed


class TestEndToEndTraining:
    """Test complete training pipeline"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = Path(self.temp_dir) / "checkpoints"
        self.log_dir = Path(self.temp_dir) / "logs"
        self.checkpoint_dir.mkdir(parents=True)
        self.log_dir.mkdir(parents=True)
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Setup single GPU environment
        os.environ['RANK'] = '0'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
    
    def teardown_method(self):
        """Cleanup after test"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        for key in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']:
            os.environ.pop(key, None)
    
    def test_complete_training_pipeline(self):
        """Test full training pipeline with all components"""
        # Create model
        model = SimpleResNet(num_classes=10)
        
        # Create trainer
        trainer = EnhancedDistributedTrainer(
            model=model,
            strategy='ddp',
            mixed_precision=False,  # Disable for CPU testing
            checkpoint_dir=str(self.checkpoint_dir),
            log_dir=str(self.log_dir),
        )
        
        # Create dummy dataset
        num_samples = 100
        dummy_data = torch.randn(num_samples, 3, 224, 224)
        dummy_labels = torch.randint(0, 10, (num_samples,))
        dataset = TensorDataset(dummy_data, dummy_labels)
        dataloader = DataLoader(dataset, batch_size=10, shuffle=False)
        
        # Create optimizer and criterion
        optimizer = optim.SGD(trainer.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        num_epochs = 2
        for epoch in range(num_epochs):
            total_loss = 0
            for step, (inputs, targets) in enumerate(dataloader):
                metrics = trainer.train_step(
                    batch=(inputs, targets),
                    optimizer=optimizer,
                    criterion=criterion,
                    step=step,
                )
                total_loss += metrics['loss']
            
            avg_loss = total_loss / len(dataloader)
            
            # Save checkpoint
            trainer.save_checkpoint(
                epoch=epoch,
                optimizer=optimizer,
                loss=avg_loss,
                is_best=(epoch == 0 or avg_loss < best_loss if epoch > 0 else True),
            )
            
            if epoch == 0:
                best_loss = avg_loss
            else:
                best_loss = min(best_loss, avg_loss)
        
        # Verify checkpoints exist
        checkpoints = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        assert len(checkpoints) == 2, f"Expected 2 checkpoints, found {len(checkpoints)}"
        
        # Verify best model exists
        best_model_path = self.checkpoint_dir / 'best_model.pt'
        assert best_model_path.exists(), "Best model not saved"
        
        # Cleanup
        trainer.cleanup()
    
    def test_checkpoint_resume(self):
        """Test checkpoint save and resume"""
        model = SimpleResNet(num_classes=10)
        
        # First training run
        trainer1 = EnhancedDistributedTrainer(
            model=model,
            strategy='ddp',
            mixed_precision=False,
            checkpoint_dir=str(self.checkpoint_dir),
        )
        
        optimizer = optim.SGD(trainer1.model.parameters(), lr=0.01)
        
        # Save checkpoint
        trainer1.save_checkpoint(
            epoch=5,
            optimizer=optimizer,
            loss=0.5,
            is_best=True,
        )
        
        # Create new trainer and load checkpoint
        model2 = SimpleResNet(num_classes=10)
        trainer2 = EnhancedDistributedTrainer(
            model=model2,
            strategy='ddp',
            mixed_precision=False,
            checkpoint_dir=str(self.checkpoint_dir),
        )
        
        optimizer2 = optim.SGD(trainer2.model.parameters(), lr=0.01)
        
        # Load checkpoint
        epoch, loss = trainer2.load_checkpoint(
            str(self.checkpoint_dir / 'checkpoint_epoch_5.pt'),
            optimizer2,
        )
        
        assert epoch == 5, f"Expected epoch 5, got {epoch}"
        assert loss == 0.5, f"Expected loss 0.5, got {loss}"
        
        # Cleanup
        trainer1.cleanup()
        trainer2.cleanup()
    
    def test_monitoring_integration(self):
        """Test monitoring integration"""
        monitor = DistributedMonitor(log_dir=str(self.log_dir))
        
        # Log some metrics
        for step in range(10):
            monitor.log_training_step(
                loss=0.5 - step * 0.01,
                batch_size=32,
                step_time=0.1,
                comm_time=0.02,
            )
        
        # Get summary
        stats = monitor.get_summary_stats()
        
        assert 'total_steps' in stats
        assert stats['total_steps'] == 10
        assert 'avg_loss' in stats
        
        # Export metrics
        output_path = self.log_dir / 'metrics.json'
        monitor.export_metrics(str(output_path))
        
        assert output_path.exists(), "Metrics not exported"
        
        # Cleanup
        monitor.close()


class TestConfigurationManagement:
    """Test configuration system"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / "configs"
        self.config_dir.mkdir(parents=True)
    
    def teardown_method(self):
        """Cleanup"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_config_creation_and_loading(self):
        """Test config save and load"""
        config_manager = ConfigManager(config_dir=str(self.config_dir))
        
        # Create default configs
        config_manager.create_default_configs()
        
        # Verify files created
        assert (self.config_dir / 'dev.yaml').exists()
        assert (self.config_dir / 'staging.yaml').exists()
        assert (self.config_dir / 'production.yaml').exists()
        
        # Load and validate
        prod_config = config_manager.load_config('production')
        
        assert prod_config.training.batch_size > 0
        assert prod_config.distributed.strategy in ['ddp', 'fsdp']
    
    def test_config_validation(self):
        """Test config validation"""
        config = ExperimentConfig()
        
        # Should pass validation
        config.validate()
        
        # Invalid config should raise error
        config.training.batch_size = -1
        
        with pytest.raises(AssertionError):
            config.validate()


class TestDataPipeline:
    """Test data loading and preprocessing"""
    
    def test_dataloader_creation(self):
        """Test creating distributed dataloader"""
        from torch.utils.data.distributed import DistributedSampler
        
        # Create dataset
        dataset = TensorDataset(
            torch.randn(100, 3, 224, 224),
            torch.randint(0, 10, (100,))
        )
        
        # Create sampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=1,
            rank=0,
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=10,
            sampler=sampler,
            num_workers=0,
        )
        
        # Verify
        assert len(dataloader) == 10
        
        for batch in dataloader:
            inputs, targets = batch
            assert inputs.shape == (10, 3, 224, 224)
            assert targets.shape == (10,)
            break


class TestProductionFeatures:
    """Test production-specific features"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_auto_batch_sizing(self):
        """Test automatic batch size selection"""
        from scripts.production_train import find_optimal_batch_size
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        model = SimpleResNet()
        device = torch.device('cuda:0')
        model = model.to(device)
        
        # Find optimal batch size
        batch_size = find_optimal_batch_size(model, device, max_batch_size=128)
        
        assert batch_size >= 2
        assert batch_size <= 128
    
    def test_graceful_shutdown(self):
        """Test graceful shutdown handling"""
        from src.monitoring.health_monitoring import GracefulShutdown
        
        # Create mock trainer
        class MockTrainer:
            def save_checkpoint(self, **kwargs):
                pass
            def cleanup(self):
                pass
        
        trainer = MockTrainer()
        shutdown_handler = GracefulShutdown(
            trainer=trainer,
            checkpoint_dir=self.temp_dir,
        )
        
        # Should not be requested initially
        assert not shutdown_handler.shutdown_requested


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
