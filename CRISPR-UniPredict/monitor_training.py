"""
Real-time training monitor for CRISPR-UniPredict
Shows live progress from logs without interrupting training
"""
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

def monitor_training():
    """Monitor training progress in real-time"""
    
    # Find latest experiment
    logs_dir = Path("logs")
    experiments = sorted([d for d in logs_dir.iterdir() if d.is_dir()], 
                        key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not experiments:
        print("❌ No experiments found")
        return
    
    latest_exp = experiments[0]
    print(f"📊 Monitoring: {latest_exp.name}")
    print("=" * 70)
    
    last_epoch = -1
    last_batch = -1
    
    while True:
        try:
            # Check if training is still running
            config_file = latest_exp / "config.json"
            if not config_file.exists():
                print("⏳ Waiting for training to start...")
                time.sleep(2)
                continue
            
            # Try to read tensorboard events (if available)
            tb_dir = latest_exp / "events"
            if tb_dir.exists():
                events = sorted(tb_dir.glob("events.out.tfevents.*"))
                if events:
                    print(f"✓ TensorBoard logging: {tb_dir}")
            
            # Check log files
            log_files = sorted(latest_exp.glob("*.log"))
            if log_files:
                latest_log = log_files[-1]
                with open(latest_log, 'r') as f:
                    lines = f.readlines()
                    
                # Show last 20 lines of progress
                print(f"\n⏰ Last updated: {datetime.now().strftime('%H:%M:%S')}")
                print("-" * 70)
                
                for line in lines[-15:]:
                    line = line.strip()
                    if line:
                        # Color code different message types
                        if "ERROR" in line:
                            print(f"❌ {line}")
                        elif "WARNING" in line:
                            print(f"⚠️  {line}")
                        elif "Epoch" in line or "Validating" in line:
                            print(f"📈 {line}")
                        elif "loss=" in line:
                            print(f"📊 {line}")
                        else:
                            print(f"   {line}")
                
                print("-" * 70)
            
            time.sleep(5)  # Update every 5 seconds
            
        except KeyboardInterrupt:
            print("\n\n✓ Monitoring stopped")
            break
        except Exception as e:
            print(f"⚠️  Monitor error: {e}")
            time.sleep(2)

if __name__ == "__main__":
    print("🚀 CRISPR-UniPredict Training Monitor")
    print("   (Press Ctrl+C to stop monitoring)")
    print()
    monitor_training()
