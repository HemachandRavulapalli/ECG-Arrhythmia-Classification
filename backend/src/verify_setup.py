#!/usr/bin/env python3
"""
verify_setup.py - Verify that the project is set up correctly
"""
import os
import sys
import json

def check_dependencies():
    """Check if required Python packages are installed"""
    print("📦 Checking Python dependencies...")
    required = [
        'numpy', 'pandas', 'scikit-learn', 'tensorflow', 
        'cv2', 'fitz', 'wfdb', 'xgboost', 'joblib'
    ]
    missing = []
    for pkg in required:
        try:
            if pkg == 'cv2':
                import cv2
            elif pkg == 'fitz':
                import fitz
            elif pkg == 'wfdb':
                import wfdb
            else:
                __import__(pkg)
            print(f"  ✅ {pkg}")
        except ImportError:
            print(f"  ❌ {pkg} - MISSING")
            missing.append(pkg)
    
    if missing:
        print(f"\n⚠️ Missing dependencies: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    return True

def check_data_structure():
    """Check if data directories exist"""
    print("\n📁 Checking data directory structure...")
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "..", "..", "data")
    
    required_dirs = [
        ("data/", data_dir),
        ("data/raw/", os.path.join(data_dir, "raw")),
        ("data/processed/", os.path.join(data_dir, "processed")),
    ]
    
    all_exist = True
    for name, path in required_dirs:
        if os.path.exists(path):
            print(f"  ✅ {name}")
        else:
            print(f"  ⚠️ {name} - Will be created automatically")
            os.makedirs(path, exist_ok=True)
    
    return True

def check_datasets():
    """Check if datasets are available"""
    print("\n📊 Checking datasets...")
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "..", "..", "data", "raw")
    
    datasets = {
        "MIT-BIH": os.path.join(data_dir, "mitdb"),
        "PTB-XL": os.path.join(data_dir, "ptbxl"),
        "Kardia": os.path.join(data_dir, "kardia"),
    }
    
    found = []
    missing = []
    
    for name, path in datasets.items():
        if os.path.exists(path) and len(os.listdir(path)) > 0:
            print(f"  ✅ {name} found")
            found.append(name)
        else:
            print(f"  ⚠️ {name} not found - synthetic data will be used")
            missing.append(name)
    
    if missing:
        print(f"\n⚠️ Missing datasets: {', '.join(missing)}")
        print("The system will generate synthetic data automatically.")
        print("For best results, download datasets from PhysioNet.")
    
    return True

def check_models():
    """Check if trained models exist"""
    print("\n🤖 Checking trained models...")
    base_dir = os.path.dirname(__file__)
    model_dir = os.path.join(base_dir, "saved_models")
    
    if not os.path.exists(model_dir):
        print("  ⚠️ No saved_models directory found")
        print("  Run training first: python3 train_pipeline.py --limit 2000 --epochs 20 --normalize")
        return False
    
    runs = [d for d in os.listdir(model_dir) if d.startswith("run_")]
    if not runs:
        print("  ⚠️ No trained models found")
        print("  Run training first: python3 train_pipeline.py --limit 2000 --epochs 20 --normalize")
        return False
    
    latest_run = max(runs, key=lambda x: os.path.getmtime(os.path.join(model_dir, x)))
    run_path = os.path.join(model_dir, latest_run)
    
    print(f"  ✅ Found {len(runs)} run(s)")
    print(f"  📂 Latest run: {latest_run}")
    
    # Check for model files
    model_files = [f for f in os.listdir(run_path) if f.endswith(('.keras', '.h5', '.joblib'))]
    if model_files:
        print(f"  ✅ Found {len(model_files)} model file(s)")
        for f in model_files[:5]:  # Show first 5
            print(f"     - {f}")
        if len(model_files) > 5:
            print(f"     ... and {len(model_files) - 5} more")
    else:
        print("  ⚠️ No model files found in latest run")
        return False
    
    # Check for classes.json
    classes_file = os.path.join(run_path, "classes.json")
    if os.path.exists(classes_file):
        with open(classes_file, 'r') as f:
            classes = json.load(f)
        print(f"  ✅ Classes file found: {len(classes)} classes")
        print(f"     {', '.join(classes)}")
    else:
        print("  ⚠️ classes.json not found (models may still work)")
    
    return True

def test_data_loading():
    """Test if data can be loaded"""
    print("\n📥 Testing data loading...")
    try:
        from data_loader import load_all_datasets
        (X_train, y_train), (X_test, y_test), classes = load_all_datasets(
            limit=100, one_hot=True, window_size=1000
        )
        print(f"  ✅ Data loaded successfully")
        print(f"     Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"     Classes: {classes}")
        return True
    except Exception as e:
        print(f"  ❌ Data loading failed: {e}")
        return False

def main():
    """Run all verification checks"""
    print("🔍 Verifying project setup...\n")
    print("=" * 60)
    
    checks = [
        ("Dependencies", check_dependencies),
        ("Data Structure", check_data_structure),
        ("Datasets", check_datasets),
        ("Models", check_models),
        ("Data Loading", test_data_loading),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"  ❌ Error checking {name}: {e}")
            results.append((name, False))
        print()
    
    print("=" * 60)
    print("\n📋 Summary:")
    
    all_passed = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {name}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All checks passed! Project is ready to use.")
    else:
        print("\n⚠️ Some checks failed. Please fix issues before proceeding.")
        print("\nQuick fixes:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Train models: python3 train_pipeline.py --limit 2000 --epochs 20 --normalize")
        print("  3. Download datasets (optional): See README.md")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
