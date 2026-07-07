"""
Comprehensive Evaluation Metrics for CRISPR-UniPredict
Computes metrics for on-target (regression) and off-target (classification) predictions
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, Union, List
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, auc, precision_recall_curve, roc_curve,
    confusion_matrix, mean_squared_error, mean_absolute_error,
    balanced_accuracy_score, average_precision_score
)
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Comprehensive metrics calculator for CRISPR-UniPredict
    
    Computes metrics for:
    - On-target prediction (regression): SCC, PCC, MAE, RMSE
    - Off-target prediction (classification): Balanced Accuracy, F1, AUROC, AUPRC
    
    Example:
        >>> calculator = MetricsCalculator()
        >>> 
        >>> # On-target metrics
        >>> on_target_metrics = calculator.compute_on_target_metrics(
        ...     predictions=on_target_pred,
        ...     targets=on_target_label
        ... )
        >>> 
        >>> # Off-target metrics
        >>> off_target_metrics = calculator.compute_off_target_metrics(
        ...     predictions=off_target_pred,
        ...     targets=off_target_label
        ... )
        >>> 
        >>> # All metrics
        >>> all_metrics = calculator.compute_all_metrics(
        ...     on_target_pred=on_target_pred,
        ...     off_target_pred=off_target_pred,
        ...     on_target_target=on_target_label,
        ...     off_target_target=off_target_label
        ... )
    """
    
    def __init__(self, epsilon: float = 1e-7):
        """
        Initialize metrics calculator
        
        Args:
            epsilon: Small value for numerical stability
        """
        self.epsilon = epsilon
    
    def compute_on_target_metrics(self,
                                  predictions: Union[np.ndarray, torch.Tensor],
                                  targets: Union[np.ndarray, torch.Tensor]) -> Dict[str, float]:
        """
        Compute on-target (regression) metrics
        
        Metrics:
        - Spearman Correlation Coefficient (SCC)
        - Pearson Correlation Coefficient (PCC)
        - Mean Absolute Error (MAE)
        - Root Mean Square Error (RMSE)
        
        Args:
            predictions: Predicted values (batch,) or (batch, 1)
            targets: Target values (batch,) or (batch, 1)
        
        Returns:
            Dictionary with metrics
        """
        # Convert to numpy
        predictions = self._to_numpy(predictions).squeeze()
        targets = self._to_numpy(targets).squeeze()
        
        # Validate
        if len(predictions) == 0:
            logger.warning("Empty predictions for on-target metrics")
            return self._empty_on_target_metrics()
        
        if len(predictions) != len(targets):
            raise ValueError(f"Length mismatch: {len(predictions)} vs {len(targets)}")
        
        metrics = {}
        
        # Spearman Correlation Coefficient
        try:
            scc, scc_pval = spearmanr(predictions, targets)
            metrics['spearman_r'] = float(scc) if not np.isnan(scc) else 0.0
            metrics['spearman_pval'] = float(scc_pval) if not np.isnan(scc_pval) else 1.0
        except Exception as e:
            logger.warning(f"Failed to compute Spearman correlation: {e}")
            metrics['spearman_r'] = 0.0
            metrics['spearman_pval'] = 1.0
        
        # Pearson Correlation Coefficient
        try:
            pcc, pcc_pval = pearsonr(predictions, targets)
            metrics['pearson_r'] = float(pcc) if not np.isnan(pcc) else 0.0
            metrics['pearson_pval'] = float(pcc_pval) if not np.isnan(pcc_pval) else 1.0
        except Exception as e:
            logger.warning(f"Failed to compute Pearson correlation: {e}")
            metrics['pearson_r'] = 0.0
            metrics['pearson_pval'] = 1.0
        
        # Mean Absolute Error
        mae = mean_absolute_error(targets, predictions)
        metrics['mae'] = float(mae)
        
        # Root Mean Square Error
        mse = mean_squared_error(targets, predictions)
        metrics['rmse'] = float(np.sqrt(mse))
        metrics['mse'] = float(mse)
        
        # Additional statistics
        metrics['pred_mean'] = float(predictions.mean())
        metrics['pred_std'] = float(predictions.std())
        metrics['target_mean'] = float(targets.mean())
        metrics['target_std'] = float(targets.std())
        
        return metrics
    
    def compute_off_target_metrics(self,
                                   predictions: Union[np.ndarray, torch.Tensor],
                                   targets: Union[np.ndarray, torch.Tensor],
                                   thresholds: Optional[List[float]] = None) -> Dict[str, Union[float, Dict]]:
        """
        Compute off-target (classification) metrics
        
        Metrics:
        - Balanced Accuracy
        - F1-score
        - AUROC (Area Under ROC Curve)
        - AUPRC (Area Under Precision-Recall Curve)
        - Precision, Recall at different thresholds
        
        Args:
            predictions: Predicted probabilities (batch,) or (batch, 1)
            targets: Target labels (batch,) or (batch, 1) with values 0 or 1
            thresholds: Thresholds to compute metrics at (default: [0.3, 0.5, 0.7])
        
        Returns:
            Dictionary with metrics
        """
        # Convert to numpy
        predictions = self._to_numpy(predictions).squeeze()
        targets = self._to_numpy(targets).squeeze().astype(int)
        
        # Validate
        if len(predictions) == 0:
            logger.warning("Empty predictions for off-target metrics")
            return self._empty_off_target_metrics()
        
        if len(predictions) != len(targets):
            raise ValueError(f"Length mismatch: {len(predictions)} vs {len(targets)}")
        
        # Clamp predictions
        predictions = np.clip(predictions, self.epsilon, 1.0 - self.epsilon)
        
        # Check if both classes present
        if len(np.unique(targets)) < 2:
            logger.warning("Only one class present in targets")
            return self._empty_off_target_metrics()
        
        metrics = {}
        
        # Binary predictions at threshold 0.5
        binary_pred = (predictions > 0.5).astype(int)
        
        # Balanced Accuracy
        try:
            balanced_acc = balanced_accuracy_score(targets, binary_pred)
            metrics['balanced_accuracy'] = float(balanced_acc)
        except Exception as e:
            logger.warning(f"Failed to compute balanced accuracy: {e}")
            metrics['balanced_accuracy'] = 0.0
        
        # F1-score
        try:
            f1 = f1_score(targets, binary_pred, zero_division=0)
            metrics['f1_score'] = float(f1)
        except Exception as e:
            logger.warning(f"Failed to compute F1-score: {e}")
            metrics['f1_score'] = 0.0
        
        # AUROC
        try:
            auroc = roc_auc_score(targets, predictions)
            metrics['auroc'] = float(auroc)
        except Exception as e:
            logger.warning(f"Failed to compute AUROC: {e}")
            metrics['auroc'] = 0.0
        
        # AUPRC
        try:
            auprc = average_precision_score(targets, predictions)
            metrics['auprc'] = float(auprc)
        except Exception as e:
            logger.warning(f"Failed to compute AUPRC: {e}")
            metrics['auprc'] = 0.0
        
        # Metrics at different thresholds
        if thresholds is None:
            thresholds = [0.3, 0.5, 0.7]
        
        metrics['threshold_metrics'] = {}
        for threshold in thresholds:
            binary_pred_th = (predictions > threshold).astype(int)
            
            try:
                precision = precision_score(targets, binary_pred_th, zero_division=0)
                recall = recall_score(targets, binary_pred_th, zero_division=0)
                accuracy = accuracy_score(targets, binary_pred_th)
                
                metrics['threshold_metrics'][f'threshold_{threshold}'] = {
                    'precision': float(precision),
                    'recall': float(recall),
                    'accuracy': float(accuracy),
                    'f1': float(2 * precision * recall / (precision + recall + self.epsilon))
                }
            except Exception as e:
                logger.warning(f"Failed to compute metrics at threshold {threshold}: {e}")
        
        # Confusion matrix
        try:
            cm = confusion_matrix(targets, binary_pred)
            metrics['tn'] = int(cm[0, 0])
            metrics['fp'] = int(cm[0, 1])
            metrics['fn'] = int(cm[1, 0])
            metrics['tp'] = int(cm[1, 1])
        except Exception as e:
            logger.warning(f"Failed to compute confusion matrix: {e}")
        
        # Additional statistics
        metrics['pred_mean'] = float(predictions.mean())
        metrics['pred_std'] = float(predictions.std())
        metrics['positive_ratio'] = float((targets == 1).mean())
        metrics['negative_ratio'] = float((targets == 0).mean())
        
        return metrics
    
    def compute_all_metrics(self,
                           on_target_pred: Optional[Union[np.ndarray, torch.Tensor]] = None,
                           off_target_pred: Optional[Union[np.ndarray, torch.Tensor]] = None,
                           on_target_target: Optional[Union[np.ndarray, torch.Tensor]] = None,
                           off_target_target: Optional[Union[np.ndarray, torch.Tensor]] = None,
                           on_target_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
                           off_target_mask: Optional[Union[np.ndarray, torch.Tensor]] = None) -> Dict:
        """
        Compute all metrics for both tasks
        
        Args:
            on_target_pred: On-target predictions
            off_target_pred: Off-target predictions
            on_target_target: On-target targets
            off_target_target: Off-target targets
            on_target_mask: Mask for valid on-target samples
            off_target_mask: Mask for valid off-target samples
        
        Returns:
            Dictionary with all metrics
        """
        all_metrics = {}
        
        # On-target metrics
        if on_target_pred is not None and on_target_target is not None:
            pred = self._to_numpy(on_target_pred).squeeze()
            target = self._to_numpy(on_target_target).squeeze()
            
            # Apply mask if provided
            if on_target_mask is not None:
                mask = self._to_numpy(on_target_mask).astype(bool)
                pred = pred[mask]
                target = target[mask]
            
            all_metrics['on_target'] = self.compute_on_target_metrics(pred, target)
        
        # Off-target metrics
        if off_target_pred is not None and off_target_target is not None:
            pred = self._to_numpy(off_target_pred).squeeze()
            target = self._to_numpy(off_target_target).squeeze()
            
            # Apply mask if provided
            if off_target_mask is not None:
                mask = self._to_numpy(off_target_mask).astype(bool)
                pred = pred[mask]
                target = target[mask]
            
            all_metrics['off_target'] = self.compute_off_target_metrics(pred, target)
        
        return all_metrics
    
    @staticmethod
    def _to_numpy(tensor: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert tensor to numpy array"""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return np.asarray(tensor)
    
    @staticmethod
    def _empty_on_target_metrics() -> Dict[str, float]:
        """Return empty on-target metrics"""
        return {
            'spearman_r': 0.0,
            'spearman_pval': 1.0,
            'pearson_r': 0.0,
            'pearson_pval': 1.0,
            'mae': 0.0,
            'rmse': 0.0,
            'mse': 0.0,
            'pred_mean': 0.0,
            'pred_std': 0.0,
            'target_mean': 0.0,
            'target_std': 0.0
        }
    
    @staticmethod
    def _empty_off_target_metrics() -> Dict:
        """Return empty off-target metrics"""
        return {
            'balanced_accuracy': 0.0,
            'f1_score': 0.0,
            'auroc': 0.0,
            'auprc': 0.0,
            'threshold_metrics': {},
            'tn': 0,
            'fp': 0,
            'fn': 0,
            'tp': 0,
            'pred_mean': 0.0,
            'pred_std': 0.0,
            'positive_ratio': 0.0,
            'negative_ratio': 0.0
        }


# Convenience functions
def compute_on_target_metrics(predictions: Union[np.ndarray, torch.Tensor],
                             targets: Union[np.ndarray, torch.Tensor]) -> Dict[str, float]:
    """Compute on-target metrics"""
    calculator = MetricsCalculator()
    return calculator.compute_on_target_metrics(predictions, targets)


def compute_off_target_metrics(predictions: Union[np.ndarray, torch.Tensor],
                              targets: Union[np.ndarray, torch.Tensor]) -> Dict[str, Union[float, Dict]]:
    """Compute off-target metrics"""
    calculator = MetricsCalculator()
    return calculator.compute_off_target_metrics(predictions, targets)


# Legacy functions for backward compatibility
def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                          y_pred_proba: np.ndarray = None) -> Dict:
    """
    Compute classification metrics (legacy function).
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
    
    Returns:
        Dictionary of metrics
    """
    calculator = MetricsCalculator()
    return calculator.compute_off_target_metrics(y_pred_proba, y_true)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Compute regression metrics (legacy function).
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Dictionary of metrics
    """
    calculator = MetricsCalculator()
    return calculator.compute_on_target_metrics(y_pred, y_true)


def compute_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Tuple:
    """
    Compute ROC curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
    
    Returns:
        Tuple of (fpr, tpr, thresholds)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    return fpr, tpr, thresholds


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Confusion matrix
    """
    return confusion_matrix(y_true, y_pred)


def compute_per_dataset_metrics(datasets: Dict, predictions: Dict) -> Dict:
    """
    Compute metrics per dataset.
    
    Args:
        datasets: Dictionary of datasets
        predictions: Dictionary of predictions
    
    Returns:
        Dictionary of per-dataset metrics
    """
    calculator = MetricsCalculator()
    results = {}
    
    for dataset_name, y_true in datasets.items():
        if dataset_name in predictions:
            y_pred = predictions[dataset_name]
            
            if len(np.unique(y_true)) == 2:  # Classification
                metrics = calculator.compute_off_target_metrics(y_pred, y_true)
            else:  # Regression
                metrics = calculator.compute_on_target_metrics(y_pred, y_true)
            
            results[dataset_name] = metrics
    
    return results
