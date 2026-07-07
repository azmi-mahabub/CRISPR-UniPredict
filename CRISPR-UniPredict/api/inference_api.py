"""
CRISPR-UniPredict Inference API
FastAPI-based API for model predictions
"""

import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
import logging
from functools import lru_cache
from datetime import datetime
import json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI(
    title="CRISPR-UniPredict API",
    description="Unified CRISPR-Cas9 on-target and off-target prediction",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== MODELS ====================

class PredictionRequest(BaseModel):
    """Single prediction request"""
    sgrna: str = Field(..., min_length=20, max_length=25, description="sgRNA sequence (20-25 bp)")
    target: Optional[str] = Field(None, description="Target sequence (optional)")
    
    @validator('sgrna')
    def validate_sgrna(cls, v):
        """Validate sgRNA sequence"""
        valid_nucleotides = set('ACGT')
        if not all(n in valid_nucleotides for n in v.upper()):
            raise ValueError("sgRNA must contain only ACGT nucleotides")
        return v.upper()
    
    @validator('target')
    def validate_target(cls, v):
        """Validate target sequence"""
        if v is None:
            return v
        valid_nucleotides = set('ACGT')
        if not all(n in valid_nucleotides for n in v.upper()):
            raise ValueError("Target must contain only ACGT nucleotides")
        return v.upper()


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    sgrnas: List[str] = Field(..., description="List of sgRNA sequences")
    targets: Optional[List[str]] = Field(None, description="List of target sequences")
    
    @validator('sgrnas')
    def validate_sgrnas(cls, v):
        """Validate sgRNA sequences"""
        if len(v) == 0:
            raise ValueError("At least one sgRNA required")
        if len(v) > 1000:
            raise ValueError("Maximum 1000 sequences per batch")
        return [s.upper() for s in v]


class OnTargetResponse(BaseModel):
    """On-target prediction response"""
    efficiency_score: float = Field(..., ge=0, le=1, description="On-target efficiency (0-1)")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    sgrna: str = Field(..., description="Input sgRNA sequence")


class OffTargetResponse(BaseModel):
    """Off-target prediction response"""
    off_target_prob: float = Field(..., ge=0, le=1, description="Off-target probability (0-1)")
    is_off_target: bool = Field(..., description="Whether sgRNA has high off-target risk")
    risk_level: str = Field(..., description="Risk level: low/medium/high")
    sgrna: str = Field(..., description="Input sgRNA sequence")


class ComprehensiveResponse(BaseModel):
    """Comprehensive prediction response"""
    on_target_score: float = Field(..., ge=0, le=1, description="On-target efficiency")
    off_target_risk: float = Field(..., ge=0, le=1, description="Off-target risk")
    comprehensive_score: float = Field(..., ge=0, le=1, description="Comprehensive score")
    recommendation: str = Field(..., description="Recommendation: Excellent/Good/Acceptable/Poor")
    sgrna: str = Field(..., description="Input sgRNA sequence")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[ComprehensiveResponse] = Field(..., description="List of predictions")
    total_processed: int = Field(..., description="Total sequences processed")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


# ==================== MODEL LOADING ====================

class ModelManager:
    """Manage model loading and predictions"""
    
    def __init__(self):
        self.model = None
        self.encoder = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_loaded = False
        self._load_attempted = False
    
    def load_model(self, checkpoint_path: str = 'models/checkpoints/best.pt') -> None:
        """Load trained model"""
        try:
            logger.info(f"Loading model...")
            
            # Import model class
            from models.crispr_unipredict import CRISPRUniPredict
            from models.encoding import SequenceEncoder
            
            # Initialize model
            self.model = CRISPRUniPredict(device=self.device)
            logger.info("✓ Model initialized")
            
            # Load checkpoint if available
            checkpoint_full_path = Path(checkpoint_path)
            if checkpoint_full_path.exists():
                try:
                    checkpoint = torch.load(checkpoint_full_path, map_location=self.device)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info("✓ Model checkpoint loaded")
                except Exception as e:
                    logger.warning(f"Could not load checkpoint: {e}. Using untrained model.")
            else:
                logger.warning(f"Checkpoint not found at {checkpoint_path}, using untrained model")
            
            # Initialize encoder
            self.encoder = SequenceEncoder(device=self.device)
            logger.info("✓ Encoder initialized")
            
            self.model.eval()
            self.model_loaded = True
            logger.info("✓ Model loaded successfully")
        
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model_loaded = False
            raise
    
    def ensure_loaded(self):
        """Ensure model is loaded, load if not already attempted"""
        if not self.model_loaded and not self._load_attempted:
            self._load_attempted = True
            try:
                self.load_model()
            except Exception as e:
                logger.warning(f"Model loading failed: {e}")
    
    def predict(self, sgrna: str, target: Optional[str] = None) -> Dict:
        """Make prediction"""
        try:
            # Ensure model is loaded
            self.ensure_loaded()
            
            if not self.model_loaded:
                raise RuntimeError("Model not loaded")
            
            # Encode input
            onehot = self.encoder.one_hot_encode(sgrna)
            label = self.encoder.label_encode(sgrna, add_start_token=False)
            
            # Add batch dimension
            onehot = onehot.unsqueeze(0).to(self.device)
            label = label.unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                on_target, off_target = self.model(onehot, label, task_type='both')
            
            on_target_score = on_target.item()
            off_target_prob = off_target.item()
            
            # Compute comprehensive score
            off_target_safety = 1.0 - off_target_prob
            comprehensive_score = on_target_score * off_target_safety
            
            return {
                'on_target_score': on_target_score,
                'off_target_prob': off_target_prob,
                'off_target_safety': off_target_safety,
                'comprehensive_score': comprehensive_score
            }
        
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise


# Initialize model manager
model_manager = ModelManager()


# ==================== STARTUP/SHUTDOWN ====================

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        model_manager.load_model()
        logger.info("✓ API startup complete")
    except Exception as e:
        logger.error(f"Startup warning: {e}")
        # Don't fail startup, allow API to run with untrained model
        logger.info("API will run with untrained model")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_manager.model_loaded,
        "device": model_manager.device,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "CRISPR-UniPredict API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "on_target": "/predict/on_target",
            "off_target": "/predict/off_target",
            "comprehensive": "/predict/comprehensive",
            "batch": "/batch_predict"
        }
    }


# ==================== PREDICTION ENDPOINTS ====================

@app.post("/predict/on_target", response_model=OnTargetResponse)
async def predict_on_target(request: PredictionRequest):
    """
    Predict on-target efficiency
    
    Returns:
    - efficiency_score: On-target efficiency (0-1)
    - confidence: Prediction confidence
    """
    try:
        # Make prediction (will ensure model is loaded)
        pred = model_manager.predict(request.sgrna, request.target)
        
        # Estimate confidence (could be improved with uncertainty estimation)
        confidence = min(0.95, 0.5 + abs(pred['on_target_score'] - 0.5))
        
        return OnTargetResponse(
            efficiency_score=pred['on_target_score'],
            confidence=confidence,
            sgrna=request.sgrna
        )
    
    except Exception as e:
        logger.error(f"On-target prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/off_target", response_model=OffTargetResponse)
async def predict_off_target(request: PredictionRequest):
    """
    Predict off-target risk
    
    Returns:
    - off_target_prob: Off-target probability (0-1)
    - is_off_target: Whether sgRNA has high off-target risk
    - risk_level: Risk level (low/medium/high)
    """
    try:
        if request.target is None:
            raise HTTPException(status_code=400, detail="Target sequence required for off-target prediction")
        
        # Make prediction (will ensure model is loaded)
        pred = model_manager.predict(request.sgrna, request.target)
        
        # Determine risk level
        off_target_prob = pred['off_target_prob']
        if off_target_prob < 0.3:
            risk_level = "low"
            is_off_target = False
        elif off_target_prob < 0.7:
            risk_level = "medium"
            is_off_target = False
        else:
            risk_level = "high"
            is_off_target = True
        
        return OffTargetResponse(
            off_target_prob=off_target_prob,
            is_off_target=is_off_target,
            risk_level=risk_level,
            sgrna=request.sgrna
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Off-target prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/comprehensive", response_model=ComprehensiveResponse)
async def predict_comprehensive(request: PredictionRequest):
    """
    Comprehensive sgRNA evaluation
    
    Returns:
    - on_target_score: On-target efficiency
    - off_target_risk: Off-target probability
    - comprehensive_score: Combined score
    - recommendation: Recommendation for use
    """
    try:
        # Make prediction (will ensure model is loaded)
        pred = model_manager.predict(request.sgrna, request.target)
        
        # Generate recommendation
        score = pred['comprehensive_score']
        if score >= 0.7:
            recommendation = "Excellent sgRNA"
        elif score >= 0.6:
            recommendation = "Good sgRNA"
        elif score >= 0.4:
            recommendation = "Acceptable sgRNA"
        else:
            recommendation = "Poor sgRNA - not recommended"
        
        return ComprehensiveResponse(
            on_target_score=pred['on_target_score'],
            off_target_risk=pred['off_target_prob'],
            comprehensive_score=pred['comprehensive_score'],
            recommendation=recommendation,
            sgrna=request.sgrna
        )
    
    except Exception as e:
        logger.error(f"Comprehensive prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """
    Batch prediction for multiple sgRNA-target pairs
    
    Returns:
    - predictions: List of comprehensive predictions
    - total_processed: Number of sequences processed
    - processing_time_ms: Total processing time
    """
    try:
        # Ensure model is loaded
        model_manager.ensure_loaded()
        
        import time
        start_time = time.time()
        
        predictions = []
        
        for i, sgrna in enumerate(request.sgrnas):
            target = request.targets[i] if request.targets else None
            
            try:
                # Make prediction
                pred = model_manager.predict(sgrna, target)
                
                # Generate recommendation
                score = pred['comprehensive_score']
                if score >= 0.7:
                    recommendation = "Excellent sgRNA"
                elif score >= 0.6:
                    recommendation = "Good sgRNA"
                elif score >= 0.4:
                    recommendation = "Acceptable sgRNA"
                else:
                    recommendation = "Poor sgRNA"
                
                predictions.append(ComprehensiveResponse(
                    on_target_score=pred['on_target_score'],
                    off_target_risk=pred['off_target_prob'],
                    comprehensive_score=pred['comprehensive_score'],
                    recommendation=recommendation,
                    sgrna=sgrna
                ))
            
            except Exception as e:
                logger.warning(f"Prediction failed for {sgrna}: {e}")
                continue
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions),
            processing_time_ms=processing_time_ms
        )
    
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== UTILITY ENDPOINTS ====================

@app.get("/model/info")
async def model_info():
    """Get model information"""
    try:
        if not model_manager.model_loaded:
            return {"status": "Model not loaded"}
        
        return {
            "model_name": "CRISPR-UniPredict",
            "device": model_manager.device,
            "model_loaded": True,
            "parameters": 1992565,
            "input_length": 23,
            "tasks": ["on_target", "off_target"],
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/docs")
async def api_docs():
    """API documentation"""
    return {
        "title": "CRISPR-UniPredict API",
        "version": "1.0.0",
        "description": "Unified CRISPR-Cas9 on-target and off-target prediction API",
        "endpoints": {
            "GET /": "API information",
            "GET /health": "Health check",
            "GET /model/info": "Model information",
            "POST /predict/on_target": "Predict on-target efficiency",
            "POST /predict/off_target": "Predict off-target risk",
            "POST /predict/comprehensive": "Comprehensive evaluation",
            "POST /batch_predict": "Batch prediction"
        },
        "swagger_ui": "/docs",
        "redoc": "/redoc"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
