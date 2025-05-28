"""
Main Application Entry Point for Hair Care AI

This module provides the main application class that can run in different modes:
- GUI mode (Streamlit)
- Command-line mode
- Batch processing mode
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, List
import json

from ..processing.analysis_pipeline import AnalysisPipeline
from ..utils.logger import logger
from ..utils.config import config


class HairCareApp:
    """
    Main application class for Hair Care AI.
    
    Provides multiple interfaces for scalp health analysis including
    GUI, command-line, and batch processing modes.
    """
    
    def __init__(self):
        """Initialize the Hair Care AI application."""
        self.pipeline = None
        self.app_name = "Hair Care AI"
        self.version = "1.0.0"
        
        logger.info(f"{self.app_name} v{self.version} initialized")
    
    def run_gui(self, **kwargs):
        """Run the application in GUI mode using Streamlit."""
        try:
            from .streamlit_app import run_streamlit_app
            
            logger.info("Starting GUI application...")
            run_streamlit_app()
            
        except ImportError as e:
            logger.error(f"Failed to import Streamlit components: {str(e)}")
            print("Error: Streamlit is required for GUI mode. Please install it with:")
            print("pip install streamlit")
            sys.exit(1)
        except Exception as e:
            logger.error(f"GUI application failed: {str(e)}")
            print(f"Error running GUI: {str(e)}")
            sys.exit(1)
    
    def run_cli(self, image_path: str, output_dir: Optional[str] = None, 
                analysis_mode: str = "comprehensive", **kwargs):
        """
        Run the application in command-line mode.
        
        Args:
            image_path: Path to the input image
            output_dir: Directory to save results
            analysis_mode: Analysis mode to use
            **kwargs: Additional configuration options
        """
        try:
            # Initialize pipeline
            self.pipeline = AnalysisPipeline()
            
            # Configure pipeline
            self.pipeline.configure_pipeline(
                analysis_mode=analysis_mode,
                auto_enhance=kwargs.get('auto_enhance', True),
                quality_threshold=kwargs.get('quality_threshold', 60)
            )
            
            # Initialize models
            print("Initializing AI models...")
            model_status = self.pipeline.initialize_models()
            
            if not any(model_status.values()):
                print("Error: Failed to initialize AI models")
                sys.exit(1)
            
            print(f"Models loaded: {[k for k, v in model_status.items() if v]}")
            
            # Validate input image
            if not Path(image_path).exists():
                print(f"Error: Image file not found: {image_path}")
                sys.exit(1)
            
            # Run analysis
            print(f"Analyzing image: {image_path}")
            results = self.pipeline.analyze_single_image(
                image_path,
                output_dir=output_dir,
                save_results=bool(output_dir)
            )
            
            if results['success']:
                self._print_cli_results(results)
                
                if output_dir:
                    print(f"\nDetailed results saved to: {output_dir}")
            else:
                print(f"Analysis failed: {results.get('error', 'Unknown error')}")
                sys.exit(1)
                
        except Exception as e:
            logger.error(f"CLI analysis failed: {str(e)}")
            print(f"Error: {str(e)}")
            sys.exit(1)
    
    def run_batch(self, input_dir: str, output_dir: str, 
                  analysis_mode: str = "comprehensive", **kwargs):
        """
        Run batch processing on multiple images.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save results
            analysis_mode: Analysis mode to use
            **kwargs: Additional configuration options
        """
        try:
            # Initialize pipeline
            self.pipeline = AnalysisPipeline()
            
            # Configure pipeline
            self.pipeline.configure_pipeline(
                analysis_mode=analysis_mode,
                auto_enhance=kwargs.get('auto_enhance', True),
                quality_threshold=kwargs.get('quality_threshold', 60)
            )
            
            # Initialize models
            print("Initializing AI models...")
            model_status = self.pipeline.initialize_models()
            
            if not any(model_status.values()):
                print("Error: Failed to initialize AI models")
                sys.exit(1)
            
            # Find image files
            input_path = Path(input_dir)
            if not input_path.exists():
                print(f"Error: Input directory not found: {input_dir}")
                sys.exit(1)
            
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            image_files = [
                str(f) for f in input_path.rglob('*') 
                if f.suffix.lower() in image_extensions
            ]
            
            if not image_files:
                print(f"No image files found in: {input_dir}")
                sys.exit(1)
            
            print(f"Found {len(image_files)} images to process")
            
            # Run batch analysis
            batch_results = self.pipeline.analyze_batch(
                image_files,
                output_dir,
                save_individual=True
            )
            
            # Print batch summary
            self._print_batch_summary(batch_results)
            
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            print(f"Error: {str(e)}")
            sys.exit(1)
    
    def _print_cli_results(self, results):
        """Print analysis results for CLI mode."""
        processed_results = results['processed_results']
        
        print("\n" + "="*60)
        print("SCALP HEALTH ANALYSIS RESULTS")
        print("="*60)
        
        # Health assessment
        health_assessment = processed_results['health_assessment']
        print(f"\nOVERALL HEALTH SCORE: {health_assessment['overall_score']:.1f}/100")
        print(f"CATEGORY: {health_assessment['category']}")
        print(f"DESCRIPTION: {health_assessment['description']}")
        
        # Detection summary
        detection_summary = processed_results['detection_summary']
        if detection_summary['total_detections'] > 0:
            print(f"\nDETECTED CONDITIONS:")
            for condition in detection_summary['conditions_detected']:
                readable_name = self._get_readable_condition_name(condition)
                count = detection_summary['class_distribution'][condition]
                print(f"  • {readable_name}: {count} detection(s)")
            
            print(f"\nTOTAL AREA AFFECTED: {detection_summary['area_affected']:.1f}%")
        else:
            print("\n✅ No scalp conditions detected!")
        
        # Top recommendations
        recommendations = processed_results['recommendations']
        if recommendations.get('immediate_actions'):
            print(f"\nIMMEDIATE ACTIONS:")
            for action in recommendations['immediate_actions'][:3]:
                print(f"  • {action}")
        
        if recommendations.get('care_routine'):
            print(f"\nCARE ROUTINE:")
            for routine in recommendations['care_routine'][:3]:
                print(f"  • {routine}")
        
        # Processing info
        print(f"\nPROCESSING TIME: {results['processing_time']:.2f} seconds")
        
        # Confidence metrics
        confidence_metrics = processed_results['confidence_metrics']
        print(f"OVERALL CONFIDENCE: {confidence_metrics['overall_confidence']:.2f}")
        
        print("="*60)
    
    def _print_batch_summary(self, batch_results):
        """Print batch processing summary."""
        print("\n" + "="*60)
        print("BATCH PROCESSING SUMMARY")
        print("="*60)
        
        print(f"Total Images: {batch_results['total_images']}")
        print(f"Successful Analyses: {batch_results['successful_analyses']}")
        print(f"Failed Analyses: {batch_results['failed_analyses']}")
        print(f"Success Rate: {(batch_results['successful_analyses'] / batch_results['total_images']) * 100:.1f}%")
        print(f"Total Processing Time: {batch_results['processing_time']:.2f} seconds")
        
        # Batch summary statistics
        summary = batch_results.get('batch_summary', {})
        if 'average_health_score' in summary:
            print(f"\nAVERAGE HEALTH SCORE: {summary['average_health_score']:.1f}")
            print(f"HEALTH SCORE RANGE: {summary['min_health_score']:.1f} - {summary['max_health_score']:.1f}")
            
            if summary.get('condition_prevalence'):
                print(f"\nCONDITION PREVALENCE:")
                for condition, count in summary['condition_prevalence'].items():
                    readable_name = self._get_readable_condition_name(condition)
                    percentage = (count / batch_results['successful_analyses']) * 100
                    print(f"  • {readable_name}: {count} images ({percentage:.1f}%)")
        
        print("="*60)
    
    def _get_readable_condition_name(self, class_name):
        """Convert class name to readable format."""
        readable_names = {
            'd': 'Dandruff',
            'o': 'Oiliness',
            's': 'Sensitivity',
            'ds': 'Dandruff + Sensitivity',
            'os': 'Oiliness + Sensitivity',
            'dss': 'Multiple Conditions'
        }
        return readable_names.get(class_name, class_name)
    
    def validate_setup(self):
        """Validate application setup and dependencies."""
        print(f"{self.app_name} v{self.version} - Setup Validation")
        print("="*50)
        
        issues = []
        warnings = []
        
        # Check model files
        model_path = Path(config.model_path)
        if not model_path.exists():
            issues.append(f"YOLO model file not found: {model_path}")
        else:
            print(f"✅ YOLO model found: {model_path}")
        
        # Check dependencies
        try:
            import torch
            print(f"✅ PyTorch: {torch.__version__}")
            
            if torch.cuda.is_available():
                print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
            else:
                warnings.append("CUDA not available - will use CPU")
                
        except ImportError:
            issues.append("PyTorch not installed")
        
        try:
            import ultralytics
            print(f"✅ Ultralytics: {ultralytics.__version__}")
        except ImportError:
            issues.append("Ultralytics not installed")
        
        try:
            import streamlit
            print(f"✅ Streamlit: {streamlit.__version__}")
        except ImportError:
            warnings.append("Streamlit not available - GUI mode disabled")
        
        # Check directories
        for dir_name in ['data', 'data/results', 'logs']:
            dir_path = Path(dir_name)
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"✅ Created directory: {dir_path}")
            else:
                print(f"✅ Directory exists: {dir_path}")
        
        # Print summary
        print("\n" + "="*50)
        if issues:
            print("❌ ISSUES FOUND:")
            for issue in issues:
                print(f"  • {issue}")
        
        if warnings:
            print("⚠️  WARNINGS:")
            for warning in warnings:
                print(f"  • {warning}")
        
        if not issues:
            print("✅ Setup validation passed!")
            return True
        else:
            print("❌ Setup validation failed!")
            return False


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="Hair Care AI - Professional Scalp Health Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run GUI application
  python -m src.gui.main_app --gui
  
  # Analyze single image
  python -m src.gui.main_app --cli --image path/to/image.jpg --output results/
  
  # Batch process images
  python -m src.gui.main_app --batch --input images/ --output results/
  
  # Validate setup
  python -m src.gui.main_app --validate
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--gui', action='store_true', help='Run GUI application')
    mode_group.add_argument('--cli', action='store_true', help='Run command-line interface')
    mode_group.add_argument('--batch', action='store_true', help='Run batch processing')
    mode_group.add_argument('--validate', action='store_true', help='Validate setup')
    
    # CLI/Batch arguments
    parser.add_argument('--image', type=str, help='Input image path (CLI mode)')
    parser.add_argument('--input', type=str, help='Input directory (batch mode)')
    parser.add_argument('--output', type=str, help='Output directory')
    
    # Configuration arguments
    parser.add_argument('--mode', type=str, default='comprehensive',
                       choices=['comprehensive', 'yolo_only', 'cnn_only'],
                       help='Analysis mode')
    parser.add_argument('--confidence', type=float, default=0.25,
                       help='Confidence threshold (0.1-0.9)')
    parser.add_argument('--no-enhance', action='store_true',
                       help='Disable automatic image enhancement')
    parser.add_argument('--quality-threshold', type=int, default=60,
                       help='Minimum image quality threshold (0-100)')
    
    args = parser.parse_args()
    
    # Create application instance
    app = HairCareApp()
    
    # Run based on mode
    if args.validate:
        success = app.validate_setup()
        sys.exit(0 if success else 1)
    
    elif args.gui:
        app.run_gui()
    
    elif args.cli:
        if not args.image:
            print("Error: --image is required for CLI mode")
            sys.exit(1)
        
        app.run_cli(
            image_path=args.image,
            output_dir=args.output,
            analysis_mode=args.mode,
            auto_enhance=not args.no_enhance,
            quality_threshold=args.quality_threshold
        )
    
    elif args.batch:
        if not args.input or not args.output:
            print("Error: --input and --output are required for batch mode")
            sys.exit(1)
        
        app.run_batch(
            input_dir=args.input,
            output_dir=args.output,
            analysis_mode=args.mode,
            auto_enhance=not args.no_enhance,
            quality_threshold=args.quality_threshold
        )


if __name__ == "__main__":
    main() 