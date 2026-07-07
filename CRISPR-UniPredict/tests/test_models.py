"""
Unit tests for CRISPR-UniPredict models and components
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile

# Import modules to test
from models.crispr_unipredict import CRISPRUniPredict
from models.encoding import SequenceEncoder
from utils.preprocessing.dataset import CRISPRDataset
from utils.evaluation.metrics import MetricsCalculator


# ==================== ENCODING TESTS ====================

class TestSequenceEncoder:
    """Test SequenceEncoder functionality"""
    
    @pytest.fixture
    def encoder(self):
        """Create encoder instance"""
        return SequenceEncoder(device='cpu')
    
    def test_one_hot_encoding_shape(self, encoder):
        """Test one-hot encoding produces correct shape"""
        seq = "ACGT"
        encoded = encoder.one_hot_encode(seq)
        assert encoded.shape == (4, 4), f"Expected shape (4, 4), got {encoded.shape}"
    
    def test_one_hot_encoding_values(self, encoder):
        """Test one-hot encoding produces correct values"""
        seq = "ACGT"
        encoded = encoder.one_hot_encode(seq)
        
        # Each position should have exactly one 1
        assert torch.sum(encoded) == 4.0, "One-hot encoding should have 4 ones total"
        
        # Each column should sum to 1
        assert torch.allclose(torch.sum(encoded, dim=0), torch.ones(4))
    
    def test_one_hot_encoding_nucleotides(self, encoder):
        """Test one-hot encoding for each nucleotide"""
        for i, nuc in enumerate(['A', 'C', 'G', 'T']):
            encoded = encoder.one_hot_encode(nuc)
            assert encoded[i, 0] == 1.0, f"Nucleotide {nuc} should map to position {i}"
    
    def test_label_encoding_shape(self, encoder):
        """Test label encoding produces correct shape"""
        seq = "ACGT"
        encoded = encoder.label_encode(seq, add_start_token=False)
        assert len(encoded) == 4, f"Expected length 4, got {len(encoded)}"
    
    def test_label_encoding_values(self, encoder):
        """Test label encoding produces correct values"""
        seq = "ACGT"
        encoded = encoder.label_encode(seq, add_start_token=False)
        
        # Check values (1-indexed for A, C, G, T)
        expected = [1, 2, 3, 4]
        assert list(encoded) == expected, f"Expected {expected}, got {list(encoded)}"
    
    def test_label_encoding_with_start_token(self, encoder):
        """Test label encoding with start token"""
        seq = "ACG"
        encoded = encoder.label_encode(seq, add_start_token=True)
        
        # Should have start token (0) + sequence
        assert len(encoded) == 4, "Should have start token + 3 nucleotides"
        assert encoded[0] == 0, "First token should be start token"
    
    def test_invalid_sequence(self, encoder):
        """Test encoding with invalid nucleotides"""
        invalid_seq = "ACGTX"
        
        with pytest.raises((ValueError, KeyError)):
            encoder.one_hot_encode(invalid_seq)
    
    def test_empty_sequence(self, encoder):
        """Test encoding empty sequence"""
        with pytest.raises((ValueError, IndexError)):
            encoder.one_hot_encode("")
    
    def test_case_insensitivity(self, encoder):
        """Test that encoding is case-insensitive"""
        seq_lower = "acgt"
        seq_upper = "ACGT"
        
        encoded_lower = encoder.one_hot_encode(seq_lower)
        encoded_upper = encoder.one_hot_encode(seq_upper)
        
        assert torch.allclose(encoded_lower, encoded_upper)


# ==================== MODEL COMPONENT TESTS ====================

class TestModelComponents:
    """Test individual model components"""
    
    def test_model_initialization(self):
        """Test model can be initialized"""
        model = CRISPRUniPredict(device='cpu')
        assert model is not None
        assert model.device == 'cpu'
    
    def test_model_parameter_count(self):
        """Test model has expected number of parameters"""
        model = CRISPRUniPredict(device='cpu')
        total_params = model.get_total_params()
        
        # Should have around 1.9M parameters
        assert total_params > 1000000, f"Expected >1M parameters, got {total_params}"
        assert total_params < 3000000, f"Expected <3M parameters, got {total_params}"
    
    def test_model_trainable_params(self):
        """Test trainable parameter count"""
        model = CRISPRUniPredict(device='cpu')
        trainable = model.get_trainable_params()
        total = model.get_total_params()
        
        assert trainable > 0, "Should have trainable parameters"
        assert trainable <= total, "Trainable should be <= total"
    
    def test_model_info(self):
        """Test model info retrieval"""
        model = CRISPRUniPredict(device='cpu')
        info = model.get_model_info()
        
        assert 'name' in info
        assert 'parameters' in info
        assert 'device' in info
        assert info['device'] == 'cpu'


# ==================== FORWARD PASS TESTS ====================

class TestForwardPass:
    """Test model forward passes"""
    
    @pytest.fixture
    def model(self):
        """Create model instance"""
        return CRISPRUniPredict(device='cpu')
    
    @pytest.fixture
    def encoder(self):
        """Create encoder instance"""
        return SequenceEncoder(device='cpu')
    
    def test_forward_pass_on_target(self, model, encoder):
        """Test forward pass for on-target prediction"""
        sgrna = "GCTAGCTAGCTAGCTAGCTAGCT"
        
        onehot = encoder.one_hot_encode(sgrna)
        label = encoder.label_encode(sgrna, add_start_token=False)
        
        onehot = onehot.unsqueeze(0)
        label = label.unsqueeze(0)
        
        with torch.no_grad():
            on_target, _ = model(onehot, label, task_type='both')
        
        assert on_target.shape == (1, 1), f"Expected shape (1, 1), got {on_target.shape}"
        assert 0 <= on_target.item() <= 1, f"Score should be in [0, 1], got {on_target.item()}"
    
    def test_forward_pass_off_target(self, model, encoder):
        """Test forward pass for off-target prediction"""
        sgrna = "GCTAGCTAGCTAGCTAGCTAGCT"
        
        onehot = encoder.one_hot_encode(sgrna)
        label = encoder.label_encode(sgrna, add_start_token=False)
        
        onehot = onehot.unsqueeze(0)
        label = label.unsqueeze(0)
        
        with torch.no_grad():
            _, off_target = model(onehot, label, task_type='both')
        
        assert off_target.shape == (1, 1), f"Expected shape (1, 1), got {off_target.shape}"
        assert 0 <= off_target.item() <= 1, f"Score should be in [0, 1], got {off_target.item()}"
    
    def test_forward_pass_batch(self, model, encoder):
        """Test forward pass with batch"""
        sgrnas = [
            "GCTAGCTAGCTAGCTAGCTAGCT",
            "ATGCATGCATGCATGCATGCATG"
        ]
        
        oneshots = []
        labels = []
        
        for sgrna in sgrnas:
            onehot = encoder.one_hot_encode(sgrna)
            label = encoder.label_encode(sgrna, add_start_token=False)
            oneshots.append(onehot)
            labels.append(label)
        
        onehot_batch = torch.stack(oneshots)
        label_batch = torch.stack(labels)
        
        with torch.no_grad():
            on_target, off_target = model(onehot_batch, label_batch, task_type='both')
        
        assert on_target.shape == (2, 1), f"Expected batch shape (2, 1), got {on_target.shape}"
        assert off_target.shape == (2, 1), f"Expected batch shape (2, 1), got {off_target.shape}"
    
    def test_no_nan_output(self, model, encoder):
        """Test that model doesn't produce NaN"""
        sgrna = "GCTAGCTAGCTAGCTAGCTAGCT"
        
        onehot = encoder.one_hot_encode(sgrna)
        label = encoder.label_encode(sgrna, add_start_token=False)
        
        onehot = onehot.unsqueeze(0)
        label = label.unsqueeze(0)
        
        with torch.no_grad():
            on_target, off_target = model(onehot, label, task_type='both')
        
        assert not torch.isnan(on_target).any(), "On-target output contains NaN"
        assert not torch.isnan(off_target).any(), "Off-target output contains NaN"
    
    def test_no_inf_output(self, model, encoder):
        """Test that model doesn't produce Inf"""
        sgrna = "GCTAGCTAGCTAGCTAGCTAGCT"
        
        onehot = encoder.one_hot_encode(sgrna)
        label = encoder.label_encode(sgrna, add_start_token=False)
        
        onehot = onehot.unsqueeze(0)
        label = label.unsqueeze(0)
        
        with torch.no_grad():
            on_target, off_target = model(onehot, label, task_type='both')
        
        assert not torch.isinf(on_target).any(), "On-target output contains Inf"
        assert not torch.isinf(off_target).any(), "Off-target output contains Inf"


# ==================== GRADIENT TESTS ====================

class TestGradients:
    """Test gradient flow"""
    
    def test_gradient_flow(self):
        """Test that gradients flow through model"""
        model = CRISPRUniPredict(device='cpu')
        encoder = SequenceEncoder(device='cpu')
        
        sgrna = "GCTAGCTAGCTAGCTAGCTAGCT"
        onehot = encoder.one_hot_encode(sgrna).unsqueeze(0)
        label = encoder.label_encode(sgrna, add_start_token=False).unsqueeze(0)
        
        onehot.requires_grad = True
        
        on_target, off_target = model(onehot, label, task_type='both')
        loss = on_target.sum() + off_target.sum()
        loss.backward()
        
        assert onehot.grad is not None, "Gradients should flow to input"
        assert onehot.grad.abs().sum() > 0, "Gradients should be non-zero"


# ==================== METRICS TESTS ====================

class TestMetricsCalculator:
    """Test metrics calculation"""
    
    @pytest.fixture
    def calculator(self):
        """Create metrics calculator"""
        return MetricsCalculator()
    
    def test_spearman_correlation(self, calculator):
        """Test Spearman correlation calculation"""
        pred = np.array([0.1, 0.5, 0.9])
        true = np.array([0.2, 0.6, 0.8])
        
        corr = calculator.compute_spearman(pred, true)
        
        assert -1 <= corr <= 1, f"Correlation should be in [-1, 1], got {corr}"
        assert corr > 0, "Positive correlation expected"
    
    def test_pearson_correlation(self, calculator):
        """Test Pearson correlation calculation"""
        pred = np.array([0.1, 0.5, 0.9])
        true = np.array([0.2, 0.6, 0.8])
        
        corr = calculator.compute_pearson(pred, true)
        
        assert -1 <= corr <= 1, f"Correlation should be in [-1, 1], got {corr}"
    
    def test_mae(self, calculator):
        """Test MAE calculation"""
        pred = np.array([0.1, 0.5, 0.9])
        true = np.array([0.2, 0.5, 0.8])
        
        mae = calculator.compute_mae(pred, true)
        
        assert mae >= 0, "MAE should be non-negative"
        assert mae < 1, "MAE should be reasonable"
    
    def test_rmse(self, calculator):
        """Test RMSE calculation"""
        pred = np.array([0.1, 0.5, 0.9])
        true = np.array([0.2, 0.5, 0.8])
        
        rmse = calculator.compute_rmse(pred, true)
        
        assert rmse >= 0, "RMSE should be non-negative"
        assert rmse < 1, "RMSE should be reasonable"
    
    def test_auroc(self, calculator):
        """Test AUROC calculation"""
        pred = np.array([0.1, 0.3, 0.7, 0.9])
        true = np.array([0, 0, 1, 1])
        
        auroc = calculator.compute_auroc(pred, true)
        
        assert 0 <= auroc <= 1, f"AUROC should be in [0, 1], got {auroc}"
        assert auroc == 1.0, "Perfect predictions should give AUROC=1"
    
    def test_f1_score(self, calculator):
        """Test F1 score calculation"""
        pred = np.array([0.1, 0.3, 0.7, 0.9])
        true = np.array([0, 0, 1, 1])
        
        f1 = calculator.compute_f1(pred, true)
        
        assert 0 <= f1 <= 1, f"F1 should be in [0, 1], got {f1}"


# ==================== DATASET TESTS ====================

class TestDataset:
    """Test dataset loading and processing"""
    
    def test_dataset_creation(self):
        """Test dataset can be created"""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("sgrna,target,on_target_label,off_target_label\n")
            f.write("GCTAGCTAGCTAGCTAGCTAGCT,ATGCATGCATGCATGCATGCATG,0.8,0.1\n")
            f.write("ATGCATGCATGCATGCATGCATG,GCTAGCTAGCTAGCTAGCTAGCT,0.7,0.2\n")
            temp_path = f.name
        
        try:
            dataset = CRISPRDataset(temp_path)
            assert len(dataset) == 2, f"Expected 2 samples, got {len(dataset)}"
        finally:
            Path(temp_path).unlink()
    
    def test_dataset_sample(self):
        """Test dataset sample format"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("sgrna,target,on_target_label,off_target_label\n")
            f.write("GCTAGCTAGCTAGCTAGCTAGCT,ATGCATGCATGCATGCATGCATG,0.8,0.1\n")
            temp_path = f.name
        
        try:
            dataset = CRISPRDataset(temp_path)
            sample = dataset[0]
            
            assert 'sgrna_sequence' in sample or 'sgrna' in sample
            assert sample is not None
        finally:
            Path(temp_path).unlink()


# ==================== INTEGRATION TESTS ====================

class TestIntegration:
    """Integration tests"""
    
    def test_end_to_end_prediction(self):
        """Test complete prediction pipeline"""
        model = CRISPRUniPredict(device='cpu')
        encoder = SequenceEncoder(device='cpu')
        
        sgrna = "GCTAGCTAGCTAGCTAGCTAGCT"
        target = "ATGCATGCATGCATGCATGCATG"
        
        # Encode
        onehot = encoder.one_hot_encode(sgrna).unsqueeze(0)
        label = encoder.label_encode(sgrna, add_start_token=False).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            on_target, off_target = model(onehot, label, task_type='both')
        
        # Verify
        assert 0 <= on_target.item() <= 1
        assert 0 <= off_target.item() <= 1
        
        # Compute comprehensive score
        comp_score = on_target.item() * (1 - off_target.item())
        assert 0 <= comp_score <= 1


# ==================== EDGE CASE TESTS ====================

class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_all_same_nucleotide(self):
        """Test sequence with all same nucleotide"""
        encoder = SequenceEncoder(device='cpu')
        
        seq = "AAAAAAAAAAAAAAAAAAAAAA"
        encoded = encoder.one_hot_encode(seq)
        
        assert encoded.shape == (4, 22)
        assert torch.sum(encoded) == 22.0
    
    def test_minimum_length_sequence(self):
        """Test minimum length sequence"""
        encoder = SequenceEncoder(device='cpu')
        
        seq = "ACGT"
        encoded = encoder.one_hot_encode(seq)
        
        assert encoded.shape == (4, 4)
    
    def test_maximum_length_sequence(self):
        """Test maximum length sequence"""
        encoder = SequenceEncoder(device='cpu')
        
        seq = "A" * 100
        encoded = encoder.one_hot_encode(seq)
        
        assert encoded.shape == (4, 100)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
