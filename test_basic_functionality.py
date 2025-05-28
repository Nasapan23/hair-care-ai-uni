#!/usr/bin/env python3
"""
Basic Functionality Test for Hair Care AI

This script tests the core components to ensure the application is working correctly.
"""

import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all core modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.utils.config import config
        print("✅ Config module imported successfully")
        
        from src.utils.logger import logger
        print("✅ Logger module imported successfully")
        
        from src.utils.helpers import validate_image_file
        print("✅ Helpers module imported successfully")
        
        from src.models.yolo_model import YOLOModel
        print("✅ YOLO model module imported successfully")
        
        from src.models.cnn_models import ScalpCNN, HairHealthCNN
        print("✅ CNN models module imported successfully")
        
        from src.models.model_manager import ModelManager
        print("✅ Model manager imported successfully")
        
        from src.processing.image_processor import ImageProcessor
        print("✅ Image processor imported successfully")
        
        from src.processing.analysis_pipeline import AnalysisPipeline
        print("✅ Analysis pipeline imported successfully")
        
        from src.processing.result_processor import ResultProcessor
        print("✅ Result processor imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        from src.utils.config import config
        
        # Test basic config access
        model_path = config.model_path
        confidence = config.confidence_threshold
        classes = config.classes
        
        print(f"✅ Model path: {model_path}")
        print(f"✅ Confidence threshold: {confidence}")
        print(f"✅ Classes loaded: {len(classes)} classes")
        
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_image_processor():
    """Test image processing functionality."""
    print("\nTesting image processor...")
    
    try:
        from src.processing.image_processor import ImageProcessor
        
        processor = ImageProcessor()
        
        # Create a test image
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Test quality assessment
        quality = processor.assess_image_quality(test_image)
        print(f"✅ Quality assessment: {quality['overall_quality']:.1f}/100")
        
        # Test preprocessing
        preprocessed = processor.preprocess_for_analysis(test_image)
        print(f"✅ Preprocessing: {preprocessed.shape}")
        
        # Test enhancement
        enhanced = processor.enhance_image(test_image, 'mild')
        print(f"✅ Enhancement: {enhanced.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Image processor test failed: {e}")
        return False

def test_model_initialization():
    """Test model initialization (without actual model files)."""
    print("\nTesting model initialization...")
    
    try:
        from src.models.model_manager import ModelManager
        
        manager = ModelManager()
        status = manager.get_model_status()
        
        print(f"✅ Model manager created")
        print(f"✅ Device: {status['device']}")
        print(f"✅ Analysis mode: {status['analysis_mode']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model initialization test failed: {e}")
        return False

def test_result_processor():
    """Test result processing functionality."""
    print("\nTesting result processor...")
    
    try:
        from src.processing.result_processor import ResultProcessor
        
        processor = ResultProcessor()
        
        # Create mock results
        mock_results = {
            'yolo_results': {
                'detections': [],
                'health_score': 85.0,
                'total_area_affected': 0.0
            },
            'combined_analysis': {
                'overall_health_score': 85.0,
                'confidence_level': 'High',
                'model_agreement': 'High'
            },
            'recommendations': {
                'immediate_actions': ['Maintain current routine'],
                'care_routine': ['Regular washing'],
                'priority_level': 'Low'
            }
        }
        
        mock_quality = {
            'overall_quality': 75,
            'quality_category': 'Good'
        }
        
        mock_scalp = {
            'extraction_success': True,
            'method_used': 'adaptive'
        }
        
        mock_metadata = {
            'original_size': (640, 640),
            'file_path': 'test.jpg'
        }
        
        # Test processing
        processed = processor.process_analysis_results(
            mock_results, mock_quality, mock_scalp, mock_metadata
        )
        
        print(f"✅ Results processed successfully")
        print(f"✅ Health score: {processed['health_assessment']['overall_score']}")
        
        # Test report generation
        report = processor.generate_summary_report(processed)
        print(f"✅ Report generated: {len(report)} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ Result processor test failed: {e}")
        return False

def test_dependencies():
    """Test that required dependencies are available."""
    print("\nTesting dependencies...")
    
    dependencies = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('ultralytics', 'Ultralytics'),
        ('cv2', 'OpenCV'),
        ('PIL', 'Pillow'),
        ('numpy', 'NumPy'),
        ('matplotlib', 'Matplotlib'),
        ('pandas', 'Pandas'),
        ('yaml', 'PyYAML')
    ]
    
    missing = []
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name} available")
        except ImportError:
            print(f"❌ {name} missing")
            missing.append(name)
    
    # Optional dependencies
    optional_deps = [
        ('streamlit', 'Streamlit (for GUI)'),
        ('plotly', 'Plotly (for visualizations)'),
        ('reportlab', 'ReportLab (for PDF reports)')
    ]
    
    for module, name in optional_deps:
        try:
            __import__(module)
            print(f"✅ {name} available")
        except ImportError:
            print(f"⚠️  {name} missing (optional)")
    
    return len(missing) == 0

def test_file_structure():
    """Test that required files and directories exist."""
    print("\nTesting file structure...")
    
    required_files = [
        'config.yaml',
        'requirements.txt',
        'src/__init__.py',
        'src/utils/__init__.py',
        'src/models/__init__.py',
        'src/processing/__init__.py',
        'src/gui/__init__.py'
    ]
    
    required_dirs = [
        'src',
        'src/utils',
        'src/models', 
        'src/processing',
        'src/gui',
        'data',
        'data/results'
    ]
    
    missing_files = []
    missing_dirs = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} missing")
            missing_files.append(file_path)
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ missing")
            missing_dirs.append(dir_path)
    
    # Check for model file
    if Path('best.pt').exists():
        print(f"✅ best.pt (YOLOv11 model)")
    else:
        print(f"⚠️  best.pt missing (required for full functionality)")
    
    return len(missing_files) == 0 and len(missing_dirs) == 0

def main():
    """Run all tests."""
    print("="*60)
    print("HAIR CARE AI - BASIC FUNCTIONALITY TEST")
    print("="*60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Dependencies", test_dependencies),
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Image Processor", test_image_processor),
        ("Model Manager", test_model_initialization),
        ("Result Processor", test_result_processor)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The application is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 