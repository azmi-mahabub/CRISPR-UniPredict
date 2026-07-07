"""
Uncertainty Estimation for CRISPR-UniPredict
Provides confidence intervals and reliability estimates for predictions
"""

import numpy as np
import torch
import logging
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from pathlib import Path
import json
import pickle

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyResult:
    """Container for uncertainty estimation results"""
    prediction: float
    uncertainty: float
    confidence_interval: Tuple[float, float]
    method: str
    is_reliable: bool
    reliability_score: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'prediction': float(self.prediction),
            'uncertainty': float(self.uncertainty),
            'ci_lower': float(self.confidence_interval[0]),
            'ci_upper': float(self.confidence_interval[1]),
            'method': self.method,
            'is_reliable': bool(self.is_reliable),
            'reliability_score': float(self.reliability_score)
        }


class UncertaintyEstimator:
    """Estimate prediction uncertainty using multiple methods"""
    
    def __init__(self,
                 model,
                 method: str = 'mc_dropout',
                 device: str = 'cuda',
                 calibration_data: Optional[np.ndarray] = None):
        """
        Initialize uncertainty estimator
        
        Args:
            model: Trained model
            method: Uncertainty method ('mc_dropout', 'ensemble', 'conformal', 'temperature')
            device: Device to use (cuda/cpu)
            calibration_data: Optional calibration data
        """
        self.model = model
        self.method = method
        self.device = device
        self.calibrated = False
        
        # Method-specific parameters
        self.temperature = 1.0
        self.nonconformity_scores = None
        self.quantile = None
        self.calibration_predictions = None
        self.calibration_targets = None
        
        # Calibrate if data provided
        if calibration_data is not None:
            self.calibrate(calibration_data)
        
        logger.info(f"UncertaintyEstimator initialized with method: {method}")
    
    def calibrate(self, val_data: Dict) -> None:
        """
        Calibrate uncertainty estimates on validation set
        
        Args:
            val_data: Dictionary with 'predictions', 'targets', 'sgrna_onehot', 'sgrna_label'
        """
        logger.info(f"Calibrating uncertainty estimator using {self.method}")
        
        if self.method == 'temperature_scaling':
            self._calibrate_temperature(val_data)
        elif self.method == 'conformal':
            self._calibrate_conformal(val_data)
        elif self.method == 'mc_dropout':
            self._calibrate_mc_dropout(val_data)
        
        self.calibrated = True
        logger.info("Calibration complete")
    
    def _calibrate_temperature(self, val_data: Dict) -> None:
        """Calibrate temperature parameter"""
        from scipy.optimize import minimize
        
        predictions = val_data.get('predictions', np.array([]))
        targets = val_data.get('targets', np.array([]))
        
        if len(predictions) == 0:
            logger.warning("No validation data for temperature calibration")
            return
        
        def nll_loss(temperature):
            """Negative log-likelihood loss"""
            scaled_preds = predictions / temperature
            # Clip to avoid numerical issues
            scaled_preds = np.clip(scaled_preds, -10, 10)
            
            # Compute cross-entropy
            ce = -np.mean(targets * np.log(scaled_preds + 1e-10) + 
                          (1 - targets) * np.log(1 - scaled_preds + 1e-10))
            return ce
        
        # Optimize temperature
        result = minimize(nll_loss, x0=1.0, bounds=[(0.1, 5.0)], method='L-BFGS-B')
        self.temperature = float(result.x[0])
        
        logger.info(f"Optimized temperature: {self.temperature:.4f}")
    
    def _calibrate_conformal(self, val_data: Dict) -> None:
        """Calibrate conformal prediction"""
        predictions = val_data.get('predictions', np.array([]))
        targets = val_data.get('targets', np.array([]))
        
        if len(predictions) == 0:
            logger.warning("No validation data for conformal calibration")
            return
        
        # Compute nonconformity scores (absolute errors)
        self.nonconformity_scores = np.abs(predictions - targets)
        
        logger.info(f"Computed {len(self.nonconformity_scores)} nonconformity scores")
    
    def _calibrate_mc_dropout(self, val_data: Dict) -> None:
        """Calibrate MC dropout"""
        # Store calibration data for later use
        self.calibration_predictions = val_data.get('predictions', np.array([]))
        self.calibration_targets = val_data.get('targets', np.array([]))
        
        logger.info(f"Stored {len(self.calibration_predictions)} calibration samples")
    
    def predict_with_uncertainty(self,
                                sgrna_onehot: torch.Tensor,
                                sgrna_label: torch.Tensor,
                                task_type: str = 'on_target',
                                n_samples: int = 100,
                                confidence_level: float = 0.95) -> UncertaintyResult:
        """
        Make prediction with uncertainty estimate
        
        Args:
            sgrna_onehot: One-hot encoded sgRNA
            sgrna_label: Label encoded sgRNA
            task_type: Task type ('on_target' or 'off_target')
            n_samples: Number of samples for MC dropout
            confidence_level: Confidence level for interval (default: 0.95)
        
        Returns:
            UncertaintyResult with prediction and uncertainty
        """
        if self.method == 'mc_dropout':
            return self._predict_mc_dropout(sgrna_onehot, sgrna_label, task_type, n_samples, confidence_level)
        elif self.method == 'temperature':
            return self._predict_temperature(sgrna_onehot, sgrna_label, task_type, confidence_level)
        elif self.method == 'conformal':
            return self._predict_conformal(sgrna_onehot, sgrna_label, task_type, confidence_level)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _predict_mc_dropout(self,
                           sgrna_onehot: torch.Tensor,
                           sgrna_label: torch.Tensor,
                           task_type: str,
                           n_samples: int,
                           confidence_level: float) -> UncertaintyResult:
        """MC dropout prediction"""
        predictions = []
        
        # Enable dropout
        self.model.train()
        
        with torch.no_grad():
            for _ in range(n_samples):
                if task_type == 'on_target':
                    pred, _ = self.model(sgrna_onehot, sgrna_label, task_type='both')
                else:
                    _, pred = self.model(sgrna_onehot, sgrna_label, task_type='both')
                
                predictions.append(pred.cpu().numpy())
        
        # Disable dropout
        self.model.eval()
        
        predictions = np.array(predictions).flatten()
        
        # Compute statistics
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        # Compute confidence interval
        z_score = self._get_z_score(confidence_level)
        ci_lower = mean_pred - z_score * std_pred
        ci_upper = mean_pred + z_score * std_pred
        
        # Reliability score (inverse of uncertainty)
        reliability_score = 1.0 / (1.0 + std_pred)
        is_reliable = std_pred < 0.1
        
        return UncertaintyResult(
            prediction=mean_pred,
            uncertainty=std_pred,
            confidence_interval=(ci_lower, ci_upper),
            method='mc_dropout',
            is_reliable=is_reliable,
            reliability_score=reliability_score
        )
    
    def _predict_temperature(self,
                            sgrna_onehot: torch.Tensor,
                            sgrna_label: torch.Tensor,
                            task_type: str,
                            confidence_level: float) -> UncertaintyResult:
        """Temperature scaling prediction"""
        self.model.eval()
        
        with torch.no_grad():
            if task_type == 'on_target':
                pred, _ = self.model(sgrna_onehot, sgrna_label, task_type='both')
            else:
                _, pred = self.model(sgrna_onehot, sgrna_label, task_type='both')
            
            pred = pred.cpu().numpy().flatten()[0]
        
        # Apply temperature scaling
        scaled_pred = pred / self.temperature
        scaled_pred = np.clip(scaled_pred, 0, 1)
        
        # Estimate uncertainty from calibration
        if self.calibration_predictions is not None:
            calibration_errors = np.abs(self.calibration_predictions - self.calibration_targets)
            uncertainty = np.std(calibration_errors)
        else:
            uncertainty = 0.1
        
        # Compute confidence interval
        z_score = self._get_z_score(confidence_level)
        ci_lower = scaled_pred - z_score * uncertainty
        ci_upper = scaled_pred + z_score * uncertainty
        
        # Reliability score
        reliability_score = 1.0 / (1.0 + uncertainty)
        is_reliable = uncertainty < 0.1
        
        return UncertaintyResult(
            prediction=scaled_pred,
            uncertainty=uncertainty,
            confidence_interval=(ci_lower, ci_upper),
            method='temperature',
            is_reliable=is_reliable,
            reliability_score=reliability_score
        )
    
    def _predict_conformal(self,
                          sgrna_onehot: torch.Tensor,
                          sgrna_label: torch.Tensor,
                          task_type: str,
                          confidence_level: float) -> UncertaintyResult:
        """Conformal prediction"""
        if self.nonconformity_scores is None:
            raise ValueError("Conformal prediction requires calibration")
        
        self.model.eval()
        
        with torch.no_grad():
            if task_type == 'on_target':
                pred, _ = self.model(sgrna_onehot, sgrna_label, task_type='both')
            else:
                _, pred = self.model(sgrna_onehot, sgrna_label, task_type='both')
            
            pred = pred.cpu().numpy().flatten()[0]
        
        # Compute quantile for confidence level
        n = len(self.nonconformity_scores)
        quantile_idx = int(np.ceil((n + 1) * (1 - confidence_level) / n))
        quantile_idx = min(quantile_idx, n - 1)
        quantile = np.sort(self.nonconformity_scores)[quantile_idx]
        
        # Compute prediction interval
        ci_lower = pred - quantile
        ci_upper = pred + quantile
        
        # Uncertainty is the interval width
        uncertainty = quantile
        
        # Reliability score
        reliability_score = 1.0 / (1.0 + uncertainty)
        is_reliable = uncertainty < 0.1
        
        return UncertaintyResult(
            prediction=pred,
            uncertainty=uncertainty,
            confidence_interval=(ci_lower, ci_upper),
            method='conformal',
            is_reliable=is_reliable,
            reliability_score=reliability_score
        )
    
    def batch_predict_with_uncertainty(self,
                                      sgrna_onehot_batch: torch.Tensor,
                                      sgrna_label_batch: torch.Tensor,
                                      task_type: str = 'on_target',
                                      n_samples: int = 100,
                                      confidence_level: float = 0.95) -> List[UncertaintyResult]:
        """
        Batch prediction with uncertainty
        
        Args:
            sgrna_onehot_batch: Batch of one-hot encoded sgRNAs
            sgrna_label_batch: Batch of label encoded sgRNAs
            task_type: Task type
            n_samples: Number of samples for MC dropout
            confidence_level: Confidence level
        
        Returns:
            List of UncertaintyResult
        """
        results = []
        
        for i in range(len(sgrna_onehot_batch)):
            result = self.predict_with_uncertainty(
                sgrna_onehot_batch[i:i+1],
                sgrna_label_batch[i:i+1],
                task_type,
                n_samples,
                confidence_level
            )
            results.append(result)
        
        return results
    
    def is_reliable(self, result: UncertaintyResult, threshold: float = 0.1) -> bool:
        """
        Check if prediction is reliable
        
        Args:
            result: UncertaintyResult
            threshold: Uncertainty threshold
        
        Returns:
            True if reliable
        """
        return result.uncertainty < threshold
    
    def get_reliability_score(self, result: UncertaintyResult) -> float:
        """
        Get reliability score (0-1)
        
        Args:
            result: UncertaintyResult
        
        Returns:
            Reliability score
        """
        return result.reliability_score
    
    def save_calibration(self, save_path: str) -> None:
        """
        Save calibration parameters
        
        Args:
            save_path: Path to save calibration
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        calibration = {
            'method': self.method,
            'temperature': self.temperature,
            'calibrated': self.calibrated
        }
        
        if self.nonconformity_scores is not None:
            calibration['nonconformity_scores'] = self.nonconformity_scores.tolist()
        
        with open(save_path, 'w') as f:
            json.dump(calibration, f, indent=2)
        
        logger.info(f"Saved calibration to {save_path}")
    
    def load_calibration(self, load_path: str) -> None:
        """
        Load calibration parameters
        
        Args:
            load_path: Path to load calibration
        """
        load_path = Path(load_path)
        
        with open(load_path, 'r') as f:
            calibration = json.load(f)
        
        self.method = calibration.get('method', self.method)
        self.temperature = calibration.get('temperature', 1.0)
        self.calibrated = calibration.get('calibrated', False)
        
        if 'nonconformity_scores' in calibration:
            self.nonconformity_scores = np.array(calibration['nonconformity_scores'])
        
        logger.info(f"Loaded calibration from {load_path}")
    
    @staticmethod
    def _get_z_score(confidence_level: float) -> float:
        """Get z-score for confidence level"""
        # Common z-scores
        z_scores = {
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576
        }
        
        if confidence_level in z_scores:
            return z_scores[confidence_level]
        
        # Approximate using inverse normal
        from scipy import stats
        return stats.norm.ppf((1 + confidence_level) / 2)


class EnsembleUncertaintyEstimator:
    """Estimate uncertainty using ensemble variance"""
    
    def __init__(self, ensemble_model, device: str = 'cuda'):
        """
        Initialize ensemble uncertainty estimator
        
        Args:
            ensemble_model: EnsembleModel instance
            device: Device to use
        """
        self.ensemble = ensemble_model
        self.device = device
        
        logger.info("EnsembleUncertaintyEstimator initialized")
    
    def predict_with_uncertainty(self,
                                sgrna_onehot: torch.Tensor,
                                sgrna_label: torch.Tensor,
                                task_type: str = 'on_target',
                                confidence_level: float = 0.95) -> UncertaintyResult:
        """
        Predict with ensemble uncertainty
        
        Args:
            sgrna_onehot: One-hot encoded sgRNA
            sgrna_label: Label encoded sgRNA
            task_type: Task type
            confidence_level: Confidence level
        
        Returns:
            UncertaintyResult
        """
        # Get predictions from each model
        predictions_dict = self.ensemble.get_model_predictions(sgrna_onehot, sgrna_label)
        
        if task_type == 'on_target':
            predictions = np.array([p[0].cpu().numpy() for p in predictions_dict.values()]).flatten()
        else:
            predictions = np.array([p[1].cpu().numpy() for p in predictions_dict.values()]).flatten()
        
        # Compute statistics
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        # Compute confidence interval
        z_score = self._get_z_score(confidence_level)
        ci_lower = mean_pred - z_score * std_pred
        ci_upper = mean_pred + z_score * std_pred
        
        # Reliability score
        reliability_score = 1.0 / (1.0 + std_pred)
        is_reliable = std_pred < 0.1
        
        return UncertaintyResult(
            prediction=mean_pred,
            uncertainty=std_pred,
            confidence_interval=(ci_lower, ci_upper),
            method='ensemble',
            is_reliable=is_reliable,
            reliability_score=reliability_score
        )
    
    def batch_predict_with_uncertainty(self,
                                      sgrna_onehot_batch: torch.Tensor,
                                      sgrna_label_batch: torch.Tensor,
                                      task_type: str = 'on_target',
                                      confidence_level: float = 0.95) -> List[UncertaintyResult]:
        """
        Batch prediction with uncertainty
        
        Args:
            sgrna_onehot_batch: Batch of one-hot encoded sgRNAs
            sgrna_label_batch: Batch of label encoded sgRNAs
            task_type: Task type
            confidence_level: Confidence level
        
        Returns:
            List of UncertaintyResult
        """
        results = []
        
        for i in range(len(sgrna_onehot_batch)):
            result = self.predict_with_uncertainty(
                sgrna_onehot_batch[i:i+1],
                sgrna_label_batch[i:i+1],
                task_type,
                confidence_level
            )
            results.append(result)
        
        return results
    
    @staticmethod
    def _get_z_score(confidence_level: float) -> float:
        """Get z-score for confidence level"""
        z_scores = {
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576
        }
        
        if confidence_level in z_scores:
            return z_scores[confidence_level]
        
        from scipy import stats
        return stats.norm.ppf((1 + confidence_level) / 2)


class UncertaintyAnalyzer:
    """Analyze and visualize uncertainty estimates"""
    
    def __init__(self, estimator: UncertaintyEstimator):
        """
        Initialize uncertainty analyzer
        
        Args:
            estimator: UncertaintyEstimator instance
        """
        self.estimator = estimator
    
    def analyze_calibration(self, val_predictions: np.ndarray, val_targets: np.ndarray) -> Dict:
        """
        Analyze calibration quality
        
        Args:
            val_predictions: Validation predictions
            val_targets: Validation targets
        
        Returns:
            Dictionary with calibration metrics
        """
        errors = np.abs(val_predictions - val_targets)
        
        metrics = {
            'mean_error': float(np.mean(errors)),
            'std_error': float(np.std(errors)),
            'max_error': float(np.max(errors)),
            'min_error': float(np.min(errors)),
            'median_error': float(np.median(errors)),
            'q25_error': float(np.percentile(errors, 25)),
            'q75_error': float(np.percentile(errors, 75))
        }
        
        return metrics
    
    def compute_coverage(self, results: List[UncertaintyResult], targets: np.ndarray) -> float:
        """
        Compute empirical coverage of confidence intervals
        
        Args:
            results: List of UncertaintyResult
            targets: Target values
        
        Returns:
            Coverage percentage
        """
        covered = 0
        
        for result, target in zip(results, targets):
            ci_lower, ci_upper = result.confidence_interval
            if ci_lower <= target <= ci_upper:
                covered += 1
        
        coverage = covered / len(results)
        return coverage
    
    def compute_interval_width(self, results: List[UncertaintyResult]) -> float:
        """
        Compute average confidence interval width
        
        Args:
            results: List of UncertaintyResult
        
        Returns:
            Average interval width
        """
        widths = [ci[1] - ci[0] for ci in [r.confidence_interval for r in results]]
        return float(np.mean(widths))
    
    def plot_uncertainty(self, results: List[UncertaintyResult], targets: np.ndarray, save_path: Optional[str] = None) -> None:
        """
        Plot predictions with uncertainty bands
        
        Args:
            results: List of UncertaintyResult
            targets: Target values
            save_path: Optional path to save figure
        """
        import matplotlib.pyplot as plt
        
        predictions = np.array([r.prediction for r in results])
        ci_lowers = np.array([r.confidence_interval[0] for r in results])
        ci_uppers = np.array([r.confidence_interval[1] for r in results])
        
        x = np.arange(len(predictions))
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot confidence intervals
        ax.fill_between(x, ci_lowers, ci_uppers, alpha=0.3, label='95% CI')
        
        # Plot predictions
        ax.plot(x, predictions, 'b-', linewidth=2, label='Prediction')
        
        # Plot targets
        ax.scatter(x, targets, color='red', s=50, alpha=0.6, label='Target')
        
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Value')
        ax.set_title('Predictions with Uncertainty Bands')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved uncertainty plot to {save_path}")
        
        plt.close()
