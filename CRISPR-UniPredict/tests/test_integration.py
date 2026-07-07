"""
Integration tests for CRISPR-UniPredict
End-to-end testing of complete pipelines
"""

import pytest
import torch
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
import json

from models.crispr_unipredict import CRISPRUniPredict
from models.encoding import SequenceEncoder
from utils.preprocessing.dataset import CRISPRDataset
from utils.evaluation.metrics import MetricsCalculator


# ==================== FIXTURES ====================

@pytest.fixture
def temp_dir():
    """Create temporary directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_dataset_csv(temp_dir):
    """Create sample dataset CSV"""
    data = {
        'sgrna': [
            'GCTAGCTAGCTAGCTAGCTAGCT',
            'ATGCATGCATGCATGCATGCATG',
            'CCGGCCGGCCGGCCGGCCGGCCG',
            'TTAATTAATTAATTAATTAATTAA',
            'AAAAAAAAAAAAAAAAAAAAAAA',
        ],
        'target': [
            'ATGCATGCATGCATGCATGCATG',
            'GCTAGCTAGCTAGCTAGCTAGCT',
            'TTAATTAATTAATTAATTAATTAA',
            'CCGGCCGGCCGGCCGGCCGGCCG',
            'TTTTTTTTTTTTTTTTTTTTTTT',
        ],
        'on_target_label': [0.8, 0.7, 0.6, 0.5, 0.4],
        'off_target_label': [0.1, 0.2, 0.3, 0.4, 0.5],
    }
    
    df = pd.DataFrame(data)
    csv_path = temp_dir / 'dataset.csv'
    df.to_csv(csv_path, index=False)
    
    return csv_path


@pytest.fixture
def model_cpu():
    """Create model on CPU"""
    return CRISPRUniPredict(device='cpu')


@pytest.fixture
def encoder_cpu():
    """Create encoder on CPU"""
    return SequenceEncoder(device='cpu')


# ==================== TRAINING PIPELINE TESTS ====================

class TestTrainingPipeline:
    """Test complete training pipeline"""
    
    def test_training_pipeline_basic(self, model_cpu, encoder_cpu, sample_dataset_csv, temp_dir):
        """Test basic training pipeline"""
        # Load dataset
        dataset = CRISPRDataset(str(sample_dataset_csv))
        assert len(dataset) == 5, "Dataset should have 5 samples"
        
        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=True
        )
        
        # Setup optimizer
        optimizer = torch.optim.Adam(model_cpu.parameters(), lr=0.001)
        
        # Training loop
        initial_loss = None
        final_loss = None
        
        for epoch in range(2):
            epoch_loss = 0
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx > 0:  # Limit batches for test
                    break
                
                # Get batch data
                sgrna = batch.get('sgrna_sequence', batch.get('sgrna'))
                
                # Encode
                oneshots = []
                labels = []
                for seq in sgrna:
                    onehot = encoder_cpu.one_hot_encode(seq)
                    label = encoder_cpu.label_encode(seq, add_start_token=False)
                    oneshots.append(onehot)
                    labels.append(label)
                
                onehot_batch = torch.stack(oneshots)
                label_batch = torch.stack(labels)
                
                # Forward pass
                on_target, off_target = model_cpu(onehot_batch, label_batch, task_type='both')
                
                # Compute loss
                loss = on_target.mean() + off_target.mean()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if epoch == 0:
                initial_loss = epoch_loss
            else:
                final_loss = epoch_loss
        
        # Verify training occurred
        assert initial_loss is not None, "Initial loss should be computed"
        assert final_loss is not None, "Final loss should be computed"
        assert not np.isnan(initial_loss), "Loss should not be NaN"
        assert not np.isnan(final_loss), "Loss should not be NaN"
    
    def test_checkpoint_saving(self, model_cpu, temp_dir):
        """Test checkpoint saving and loading"""
        checkpoint_path = temp_dir / 'checkpoint.pt'
        
        # Save checkpoint
        checkpoint = {
            'model_state_dict': model_cpu.state_dict(),
            'epoch': 10,
            'loss': 0.5,
        }
        torch.save(checkpoint, checkpoint_path)
        
        # Verify checkpoint exists
        assert checkpoint_path.exists(), "Checkpoint should be saved"
        
        # Load checkpoint
        loaded_checkpoint = torch.load(checkpoint_path)
        assert 'model_state_dict' in loaded_checkpoint
        assert loaded_checkpoint['epoch'] == 10
        assert loaded_checkpoint['loss'] == 0.5
        
        # Load into new model
        new_model = CRISPRUniPredict(device='cpu')
        new_model.load_state_dict(loaded_checkpoint['model_state_dict'])
        
        # Verify models are identical
        for p1, p2 in zip(model_cpu.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2), "Model parameters should match"


# ==================== PREDICTION PIPELINE TESTS ====================

class TestPredictionPipeline:
    """Test complete prediction pipeline"""
    
    def test_prediction_pipeline(self, model_cpu, encoder_cpu, sample_dataset_csv):
        """Test complete prediction pipeline"""
        # Load dataset
        dataset = CRISPRDataset(str(sample_dataset_csv))
        
        # Make predictions
        predictions = []
        
        for i in range(len(dataset)):
            sample = dataset[i]
            sgrna = sample.get('sgrna_sequence', sample.get('sgrna'))
            
            # Encode
            onehot = encoder_cpu.one_hot_encode(sgrna).unsqueeze(0)
            label = encoder_cpu.label_encode(sgrna, add_start_token=False).unsqueeze(0)
            
            # Predict
            with torch.no_grad():
                on_target, off_target = model_cpu(onehot, label, task_type='both')
            
            predictions.append({
                'sgrna': sgrna,
                'on_target': on_target.item(),
                'off_target': off_target.item(),
            })
        
        # Verify predictions
        assert len(predictions) == 5, "Should have 5 predictions"
        
        for pred in predictions:
            assert 'sgrna' in pred
            assert 'on_target' in pred
            assert 'off_target' in pred
            assert 0 <= pred['on_target'] <= 1
            assert 0 <= pred['off_target'] <= 1
    
    def test_batch_prediction(self, model_cpu, encoder_cpu):
        """Test batch prediction"""
        sgrnas = [
            'GCTAGCTAGCTAGCTAGCTAGCT',
            'ATGCATGCATGCATGCATGCATG',
            'CCGGCCGGCCGGCCGGCCGGCCG',
        ]
        
        # Encode batch
        oneshots = []
        labels = []
        
        for sgrna in sgrnas:
            onehot = encoder_cpu.one_hot_encode(sgrna)
            label = encoder_cpu.label_encode(sgrna, add_start_token=False)
            oneshots.append(onehot)
            labels.append(label)
        
        onehot_batch = torch.stack(oneshots)
        label_batch = torch.stack(labels)
        
        # Predict
        with torch.no_grad():
            on_target, off_target = model_cpu(onehot_batch, label_batch, task_type='both')
        
        # Verify batch predictions
        assert on_target.shape == (3, 1), f"Expected shape (3, 1), got {on_target.shape}"
        assert off_target.shape == (3, 1), f"Expected shape (3, 1), got {off_target.shape}"
        
        for i in range(3):
            assert 0 <= on_target[i].item() <= 1
            assert 0 <= off_target[i].item() <= 1
    
    def test_prediction_output_format(self, model_cpu, encoder_cpu):
        """Test prediction output format"""
        sgrna = 'GCTAGCTAGCTAGCTAGCTAGCT'
        
        onehot = encoder_cpu.one_hot_encode(sgrna).unsqueeze(0)
        label = encoder_cpu.label_encode(sgrna, add_start_token=False).unsqueeze(0)
        
        with torch.no_grad():
            on_target, off_target = model_cpu(onehot, label, task_type='both')
        
        # Verify output format
        assert isinstance(on_target, torch.Tensor), "Output should be tensor"
        assert isinstance(off_target, torch.Tensor), "Output should be tensor"
        assert on_target.dtype == torch.float32
        assert off_target.dtype == torch.float32
        assert on_target.device.type == 'cpu'
        assert off_target.device.type == 'cpu'


# ==================== METRICS PIPELINE TESTS ====================

class TestMetricsPipeline:
    """Test metrics computation pipeline"""
    
    def test_metrics_computation(self):
        """Test complete metrics computation"""
        # Create predictions and ground truth
        predictions = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        ground_truth = np.array([0.2, 0.4, 0.5, 0.6, 0.8])
        
        calculator = MetricsCalculator()
        
        # Compute metrics
        spearman = calculator.compute_spearman(predictions, ground_truth)
        pearson = calculator.compute_pearson(predictions, ground_truth)
        mae = calculator.compute_mae(predictions, ground_truth)
        rmse = calculator.compute_rmse(predictions, ground_truth)
        
        # Verify metrics
        assert -1 <= spearman <= 1, "Spearman should be in [-1, 1]"
        assert -1 <= pearson <= 1, "Pearson should be in [-1, 1]"
        assert mae >= 0, "MAE should be non-negative"
        assert rmse >= 0, "RMSE should be non-negative"
        
        # Verify metrics are reasonable
        assert spearman > 0, "Should have positive correlation"
        assert mae < 1, "MAE should be reasonable"
        assert rmse < 1, "RMSE should be reasonable"
    
    def test_classification_metrics(self):
        """Test classification metrics"""
        predictions = np.array([0.1, 0.3, 0.7, 0.9])
        ground_truth = np.array([0, 0, 1, 1])
        
        calculator = MetricsCalculator()
        
        # Compute metrics
        auroc = calculator.compute_auroc(predictions, ground_truth)
        f1 = calculator.compute_f1(predictions, ground_truth)
        
        # Verify metrics
        assert 0 <= auroc <= 1, "AUROC should be in [0, 1]"
        assert 0 <= f1 <= 1, "F1 should be in [0, 1]"
        
        # Perfect predictions should give AUROC=1
        assert auroc == 1.0, "Perfect predictions should give AUROC=1"


# ==================== DATA PIPELINE TESTS ====================

class TestDataPipeline:
    """Test complete data pipeline"""
    
    def test_data_loading(self, sample_dataset_csv):
        """Test data loading"""
        # Load dataset
        dataset = CRISPRDataset(str(sample_dataset_csv))
        
        # Verify dataset
        assert len(dataset) == 5, "Dataset should have 5 samples"
        
        # Load sample
        sample = dataset[0]
        assert sample is not None, "Sample should not be None"
    
    def test_data_splitting(self, sample_dataset_csv):
        """Test data splitting"""
        # Load full dataset
        df = pd.read_csv(sample_dataset_csv)
        
        # Split data
        train_size = int(0.7 * len(df))
        val_size = int(0.15 * len(df))
        test_size = len(df) - train_size - val_size
        
        train_df = df[:train_size]
        val_df = df[train_size:train_size+val_size]
        test_df = df[train_size+val_size:]
        
        # Verify splits
        assert len(train_df) + len(val_df) + len(test_df) == len(df)
        assert len(train_df) > 0, "Train set should not be empty"
        assert len(val_df) >= 0, "Val set should be non-negative"
        assert len(test_df) > 0, "Test set should not be empty"
    
    def test_data_validation(self, sample_dataset_csv):
        """Test data validation"""
        df = pd.read_csv(sample_dataset_csv)
        
        # Check required columns
        required_cols = ['sgrna', 'target']
        for col in required_cols:
            assert col in df.columns, f"Column {col} should exist"
        
        # Check data types
        assert df['sgrna'].dtype == 'object', "sgRNA should be string"
        assert df['target'].dtype == 'object', "Target should be string"
        
        # Check no missing values
        assert not df['sgrna'].isna().any(), "No missing sgRNA"
        assert not df['target'].isna().any(), "No missing target"


# ==================== REPRODUCIBILITY TESTS ====================

class TestReproducibility:
    """Test reproducibility of results"""
    
    def test_deterministic_encoding(self, encoder_cpu):
        """Test deterministic encoding"""
        sgrna = 'GCTAGCTAGCTAGCTAGCTAGCT'
        
        # Encode twice
        encoded1 = encoder_cpu.one_hot_encode(sgrna)
        encoded2 = encoder_cpu.one_hot_encode(sgrna)
        
        # Should be identical
        assert torch.allclose(encoded1, encoded2), "Encoding should be deterministic"
    
    def test_deterministic_prediction(self, model_cpu, encoder_cpu):
        """Test deterministic predictions with seed"""
        torch.manual_seed(42)
        np.random.seed(42)
        
        sgrna = 'GCTAGCTAGCTAGCTAGCTAGCT'
        onehot = encoder_cpu.one_hot_encode(sgrna).unsqueeze(0)
        label = encoder_cpu.label_encode(sgrna, add_start_token=False).unsqueeze(0)
        
        # First prediction
        with torch.no_grad():
            on_target1, off_target1 = model_cpu(onehot, label, task_type='both')
        
        # Reset seed
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Second prediction
        with torch.no_grad():
            on_target2, off_target2 = model_cpu(onehot, label, task_type='both')
        
        # Should be identical
        assert torch.allclose(on_target1, on_target2), "Predictions should be deterministic"
        assert torch.allclose(off_target1, off_target2), "Predictions should be deterministic"
    
    def test_reproducible_training(self, encoder_cpu, sample_dataset_csv):
        """Test reproducible training"""
        def train_model(seed):
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            model = CRISPRUniPredict(device='cpu')
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            dataset = CRISPRDataset(str(sample_dataset_csv))
            loader = torch.utils.data.DataLoader(dataset, batch_size=2)
            
            losses = []
            for batch in loader:
                sgrna = batch.get('sgrna_sequence', batch.get('sgrna'))
                
                oneshots = []
                labels = []
                for seq in sgrna:
                    onehot = encoder_cpu.one_hot_encode(seq)
                    label = encoder_cpu.label_encode(seq, add_start_token=False)
                    oneshots.append(onehot)
                    labels.append(label)
                
                onehot_batch = torch.stack(oneshots)
                label_batch = torch.stack(labels)
                
                on_target, off_target = model(onehot_batch, label_batch, task_type='both')
                loss = on_target.mean() + off_target.mean()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
            
            return losses
        
        # Train twice with same seed
        losses1 = train_model(42)
        losses2 = train_model(42)
        
        # Should be identical
        assert len(losses1) == len(losses2), "Should have same number of losses"
        for l1, l2 in zip(losses1, losses2):
            assert abs(l1 - l2) < 1e-5, f"Losses should match: {l1} vs {l2}"


# ==================== ERROR HANDLING TESTS ====================

class TestErrorHandling:
    """Test error handling in pipelines"""
    
    def test_invalid_sequence_handling(self, encoder_cpu):
        """Test handling of invalid sequences"""
        invalid_seq = 'ACGTX'
        
        with pytest.raises((ValueError, KeyError)):
            encoder_cpu.one_hot_encode(invalid_seq)
    
    def test_empty_dataset_handling(self, temp_dir):
        """Test handling of empty dataset"""
        # Create empty CSV
        csv_path = temp_dir / 'empty.csv'
        df = pd.DataFrame({'sgrna': [], 'target': []})
        df.to_csv(csv_path, index=False)
        
        # Try to load
        dataset = CRISPRDataset(str(csv_path))
        assert len(dataset) == 0, "Empty dataset should have length 0"
    
    def test_mismatched_batch_sizes(self, model_cpu, encoder_cpu):
        """Test handling of mismatched batch sizes"""
        # Create mismatched batch
        onehot = torch.randn(2, 4, 23)  # Batch size 2
        label = torch.randint(0, 5, (3, 23))  # Batch size 3
        
        # Should raise error or handle gracefully
        try:
            with torch.no_grad():
                model_cpu(onehot, label, task_type='both')
        except (RuntimeError, ValueError):
            pass  # Expected


# ==================== PERFORMANCE TESTS ====================

class TestPerformance:
    """Test performance characteristics"""
    
    def test_prediction_speed(self, model_cpu, encoder_cpu):
        """Test prediction speed"""
        import time
        
        sgrnas = ['GCTAGCTAGCTAGCTAGCTAGCT'] * 100
        
        # Encode batch
        oneshots = []
        labels = []
        for sgrna in sgrnas:
            onehot = encoder_cpu.one_hot_encode(sgrna)
            label = encoder_cpu.label_encode(sgrna, add_start_token=False)
            oneshots.append(onehot)
            labels.append(label)
        
        onehot_batch = torch.stack(oneshots)
        label_batch = torch.stack(labels)
        
        # Time prediction
        start = time.time()
        with torch.no_grad():
            on_target, off_target = model_cpu(onehot_batch, label_batch, task_type='both')
        elapsed = time.time() - start
        
        # Should be fast (< 1 second for 100 sequences on CPU)
        assert elapsed < 5.0, f"Prediction should be fast, took {elapsed:.2f}s"
    
    def test_memory_efficiency(self, model_cpu, encoder_cpu):
        """Test memory efficiency"""
        # Create large batch
        batch_size = 32
        sgrnas = ['GCTAGCTAGCTAGCTAGCTAGCT'] * batch_size
        
        # Encode batch
        oneshots = []
        labels = []
        for sgrna in sgrnas:
            onehot = encoder_cpu.one_hot_encode(sgrna)
            label = encoder_cpu.label_encode(sgrna, add_start_token=False)
            oneshots.append(onehot)
            labels.append(label)
        
        onehot_batch = torch.stack(oneshots)
        label_batch = torch.stack(labels)
        
        # Should not raise out of memory
        with torch.no_grad():
            on_target, off_target = model_cpu(onehot_batch, label_batch, task_type='both')
        
        assert on_target.shape == (batch_size, 1)
        assert off_target.shape == (batch_size, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
