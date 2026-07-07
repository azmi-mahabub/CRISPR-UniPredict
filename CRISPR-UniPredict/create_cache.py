#!/usr/bin/env python
"""
Create encoding cache for CRISPR-UniPredict
This is a one-time operation that pre-encodes all sequences for faster loading
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from configs.config_loader import ConfigLoader
from utils.preprocessing.dataloader_factory import DataLoaderFactory
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    print('=' * 80)
    print('PHASE 1: CREATING ENCODING CACHE')
    print('=' * 80)
    print('This will take 20-30 minutes on first run')
    print('Subsequent runs will load from cache in <5 seconds')
    print('=' * 80)
    print()
    
    try:
        # Load configuration
        logger.info("Loading configuration from configs/model_config.yaml")
        config = ConfigLoader('configs/model_config.yaml').config
        logger.info("✓ Configuration loaded successfully")
        
        # Initialize factory
        logger.info("Initializing DataLoader factory")
        factory = DataLoaderFactory(config)
        logger.info("✓ DataLoader factory initialized")
        
        # Create dataloaders (this creates cache)
        logger.info("Creating dataloaders and encoding cache...")
        logger.info("This may take 20-30 minutes...")
        dataloaders = factory.create_dataloaders()
        
        print()
        print('=' * 80)
        print('✓ CACHE CREATED SUCCESSFULLY!')
        print('=' * 80)
        print('Cache location: .cache/')
        print('  - .cache/train/encodings_*.pkl')
        print('  - .cache/val/encodings_*.pkl')
        print('  - .cache/test/encodings_*.pkl')
        print()
        print('Ready for training!')
        print('=' * 80)
        
        return 0
        
    except Exception as e:
        print()
        print('=' * 80)
        print('✗ ERROR CREATING CACHE')
        print('=' * 80)
        logger.error(f"Error: {e}", exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main())
