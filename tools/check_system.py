#!/usr/bin/env python3
"""
System Resource Checker for Master BIDS Pipeline.

Checks available hardware, GPU configuration, and recommends optimal pipeline settings.
"""

import argparse
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
from pathlib import Path


def check_python_version():
    """Check Python version."""
    ver = sys.version_info
    min_ver = (3, 10)
    
    if (ver.major, ver.minor) >= min_ver:
        print(f"✓ Python {ver.major}.{ver.minor}.{ver.micro} (OK)")
        return True
    else:
        print(f"✗ Python {ver.major}.{ver.minor}.{ver.micro} (need >= 3.10)")
        return False


def check_gpu_availability():
    """Check NVIDIA GPU availability."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total"],
            "-h", capture_output=True, text=True, timeout=5
        )
        
        if result.returncode != 0:
            print("✗ nvidia-smi not found or NVIDIA driver not installed")
            return []
        
        devices = []
        for line in result.stdout.strip().split("\n")[1:]:
            if line.strip():
                parts = [p.strip() for p in line.split(",")]
                devices.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "driver": parts[2],
                    "memory": parts[3],
                })
        
        if devices:
            print(f"✓ NVIDIA GPU(s) detected: {len(devices)} device(s)")
            for dev in devices:
                print(f"  [{dev['index']}] {dev['name']} ({dev['memory']}) "
                      f"Driver: {dev['driver']}")
            return devices
        else:
            print("✗ No NVIDIA GPUs found")
            return []
    
    except FileNotFoundError:
        print("✗ nvidia-smi not found (NVIDIA driver not installed?)")
        return []
    except Exception as e:
        print(f"⚠ GPU detection failed: {e}")
        return []


def check_cuda_availability():
    """Check PyTorch CUDA support."""
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  Device count: {torch.cuda.device_count()}")
            print(f"  CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠ PyTorch not compiled with CUDA (CPU-only mode)")
            return False
    except ImportError:
        print("✗ PyTorch not installed: pip install torch--index-url https://download.pytorch.org/whl/cu118")
        return False


def check_mediapipe():
    """Check MediaPipe availability."""
    try:
        import mediapipe
        print(f"✓ MediaPipe {mediapipe.__version__}")
        return True
    except ImportError:
        print("✗ MediaPipe not installed: pip install mediapipe")
        return False


def check_cv2():
    """Check OpenCV availability."""
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
        
        # Check CUDA support
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            print(f"  CUDA support: Yes ({cv2.cuda.getCudaEnabledDeviceCount()} devices)")
        else:
            print(f"  CUDA support: No (CPU-only)")
        
        return True
    except ImportError:
        print("✗ OpenCV not installed: pip install opencv-python")
        return False


def check_dependencies():
    """Check required dependencies."""
    required = ["pydantic", "pandas", "numpy", "scipy"]
    packages = {}
    
    for pkg in required:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            print(f"✓ {pkg} {ver}")
            packages[pkg] = True
        except ImportError:
            print(f"✗ {pkg} not installed")
            packages[pkg] = False
    
    return all(packages.values())


def check_disk_space(output_dir: Path):
    """Check available disk space."""
    try:
        stat = shutil.disk_usage(output_dir)
        available_gb = stat.free / (1024**3)
        required_gb = 2000  # Conservative estimate for 27 sessions
        
        print(f"✓ Disk space available: {available_gb:.1f} GB")
        
        if available_gb >= required_gb:
            print(f"  Sufficient for processing (~{required_gb}GB recommended)")
            return True
        else:
            print(f"  ⚠ May be insufficient (need ~{required_gb}GB for full run)")
            return False
    except Exception as e:
        print(f"⚠ Could not check disk space: {e}")
        return True


def check_data_inventory(data_dir: Path):
    """Check data inventory files."""
    required_files = [
        "high_level_data_inventory.json",
        "high_level_group_inventory.csv",
        "high_level_session_inventory.csv",
    ]
    
    inventory_dir = data_dir / "data"
    
    all_exist = True
    for fname in required_files:
        fpath = inventory_dir / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / (1024**2)
            print(f"✓ {fname} ({size_mb:.1f} MB)")
        else:
            print(f"✗ {fname} not found")
            all_exist = False
    
    return all_exist


def recommend_settings(gpu_count: int, cpu_count: int, gpu_available: bool):
    """Recommend pipeline settings based on hardware."""
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    if not gpu_available:
        workers = min(4, cpu_count // 2)
        preset = "quick"
        duration = "1-2 hours"
        
        print(f"Mode: CPU-only (no GPU)")
        print(f"  Recommended workers: {workers}")
        print(f"  Recommended preset: {preset}")
        print(f"  Estimated duration: {duration}")
        print(f"\nCommand:")
        print(f"  python tools/run_pipeline.py --preset quick --max-workers {workers}")
    else:
        if gpu_count == 1:
            workers = min(4, cpu_count // 2)
            preset = "standard"
            duration = "1-2 hours per session"
            cmd = f"python tools/run_pipeline.py --preset standard --max-workers {workers} --gpu-devices 0"
        elif gpu_count == 2:
            workers = 8
            preset = "dual_gpu"
            duration = "45 minutes per session"
            cmd = f"python tools/run_pipeline.py --preset dual_gpu --max-workers {workers} --gpu-devices 0 1"
        else:  # 3+
            workers = min(gpu_count * 2, cpu_count // 2)
            preset = "full"
            duration = "30-45 minutes per session"
            gpu_str = " ".join(str(i) for i in range(gpu_count))
            cmd = f"python tools/run_pipeline.py --preset full --max-workers {workers} --gpu-devices {gpu_str}"
        
        print(f"Mode: GPU-accelerated ({gpu_count} GPU(s))")
        print(f"  Recommended workers: {workers}")
        print(f"  Recommended preset: {preset}")
        print(f"  Estimated duration: {duration}")
        print(f"\nCommand:")
        print(f"  {cmd}")


def main():
    parser = argparse.ArgumentParser(
        description="Check system resources and recommend pipeline settings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Data directory for inventory verification",
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for disk space verification",
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("SYSTEM RESOURCE CHECK")
    print("=" * 70 + "\n")
    
    results = {
        "python": check_python_version(),
        "gpu": check_gpu_availability(),
        "cuda": check_cuda_availability(),
        "mediapipe": check_mediapipe(),
        "opencv": check_cv2(),
        "dependencies": check_dependencies(),
    }
    
    if args.data_dir:
        print("\n" + "=" * 70)
        print("DATA INVENTORY")
        print("=" * 70 + "\n")
        results["data_inventory"] = check_data_inventory(args.data_dir)
    
    if args.output_dir:
        print("\n" + "=" * 70)
        print("DISK SPACE")
        print("=" * 70 + "\n")
        results["disk_space"] = check_disk_space(args.output_dir)
    
    # Recommendations
    gpu_count = len(results["gpu"]) if isinstance(results["gpu"], list) else 0
    cpu_count = multiprocessing.cpu_count()
    gpu_available = results["cuda"]
    
    recommend_settings(gpu_count, cpu_count, gpu_available)
    
    # JSON output
    if args.json:
        output = {
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "results": {
                "python_ok": results["python"],
                "gpu_count": gpu_count,
                "cuda_available": gpu_available,
                "mediapipe": results["mediapipe"],
                "opencv": results["opencv"],
                "dependencies_ok": results["dependencies"],
            },
            "hardware": {
                "cpu_count": cpu_count,
                "gpu_count": gpu_count,
            }
        }
        print("\n" + json.dumps(output, indent=2))
    
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
