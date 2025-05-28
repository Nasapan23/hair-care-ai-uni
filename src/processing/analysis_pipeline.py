"""
Analysis Pipeline for Hair Care AI Application

This module orchestrates the complete analysis workflow from image input
to final results, coordinating between image processing, model inference,
and result generation.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable
import numpy as np
from PIL import Image

from .image_processor import ImageProcessor
from .result_processor import ResultProcessor
from ..models.model_manager import ModelManager
from ..utils.logger import logger
from ..utils.config import config
from ..utils.helpers import ensure_dir, create_output_filename


class AnalysisPipeline:
    """
    Complete analysis pipeline for scalp health assessment.
    
    Orchestrates the workflow from image input through processing,
    model inference, and result generation.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the analysis pipeline.
        
        Args:
            device: Device to run models on ('cpu', 'cuda', 'mps')
        """
        self.device = device
        
        # Initialize components
        self.image_processor = ImageProcessor()
        self.model_manager = ModelManager(device)
        self.result_processor = ResultProcessor()
        
        # Pipeline settings
        self.auto_enhance = True
        self.save_intermediate = False
        self.quality_threshold = 60  # Minimum quality score for processing
        
        # Progress tracking
        self.progress_callback = None
        self.current_step = 0
        self.total_steps = 0
        
        logger.info("AnalysisPipeline initialized")
    
    def set_progress_callback(self, callback: Callable[[int, int, str], None]):
        """
        Set callback function for progress updates.
        
        Args:
            callback: Function that takes (current_step, total_steps, description)
        """
        self.progress_callback = callback
    
    def _update_progress(self, step: int, description: str):
        """Update progress if callback is set."""
        self.current_step = step
        if self.progress_callback:
            self.progress_callback(self.current_step, self.total_steps, description)
    
    def initialize_models(self, yolo_path: Optional[str] = None,
                         scalp_cnn_path: Optional[str] = None,
                         hair_cnn_path: Optional[str] = None) -> Dict[str, bool]:
        """
        Initialize all AI models.
        
        Args:
            yolo_path: Path to YOLO model
            scalp_cnn_path: Path to ScalpCNN weights
            hair_cnn_path: Path to HairHealthCNN weights
            
        Returns:
            Dict with initialization status for each model
        """
        logger.info("Initializing AI models...")
        
        results = {
            'yolo': False,
            'scalp_cnn': False,
            'hair_cnn': False,
            'ensemble': False
        }
        
        try:
            # Load YOLO model
            results['yolo'] = self.model_manager.load_yolo_model(yolo_path)
            
            # Load CNN models
            cnn_results = self.model_manager.load_cnn_models(
                scalp_cnn_path, hair_cnn_path
            )
            results.update(cnn_results)
            
            # Create ensemble if both CNNs are available
            if results['scalp_cnn'] and results['hair_cnn']:
                results['ensemble'] = self.model_manager.create_ensemble_model()
            
            logger.info(f"Model initialization completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            return results
    
    def analyze_single_image(self, image_input: Union[str, np.ndarray, Image.Image],
                           output_dir: Optional[str] = None,
                           save_results: bool = True) -> Dict:
        """
        Analyze a single image through the complete pipeline.
        
        Args:
            image_input: Input image (path, numpy array, or PIL Image)
            output_dir: Directory to save results
            save_results: Whether to save analysis results
            
        Returns:
            Complete analysis results
        """
        self.total_steps = 8
        self.current_step = 0
        
        start_time = time.time()
        
        try:
            # Step 1: Load and validate image
            self._update_progress(1, "Loading and validating image...")
            
            if isinstance(image_input, str):
                loaded_image = self.image_processor.load_image(image_input)
                image_array = loaded_image['image']
                image_metadata = loaded_image['metadata']
            else:
                if isinstance(image_input, Image.Image):
                    image_array = np.array(image_input)
                else:
                    image_array = image_input
                
                image_metadata = {
                    'array_shape': image_array.shape,
                    'file_path': None,
                    'original_size': (image_array.shape[1], image_array.shape[0])
                }
            
            # Step 2: Assess image quality
            self._update_progress(2, "Assessing image quality...")
            quality_assessment = self.image_processor.assess_image_quality(image_array)
            
            # Check if quality meets threshold
            if quality_assessment['overall_quality'] < self.quality_threshold:
                logger.warning(f"Image quality below threshold: {quality_assessment['overall_quality']}")
                if not self.auto_enhance:
                    return {
                        'success': False,
                        'error': 'Image quality too low for analysis',
                        'quality_assessment': quality_assessment,
                        'recommendations': quality_assessment['recommendations']
                    }
            
            # Step 3: Enhance image if needed
            self._update_progress(3, "Enhancing image quality...")
            
            if self.auto_enhance and quality_assessment['overall_quality'] < 80:
                enhanced_image = self.image_processor.enhance_image(
                    image_array, 
                    enhancement_level='moderate'
                )
                logger.info("Image enhanced for better analysis")
            else:
                enhanced_image = image_array
            
            # Step 4: Preprocess for analysis
            self._update_progress(4, "Preprocessing image for analysis...")
            preprocessed_image = self.image_processor.preprocess_for_analysis(
                enhanced_image, normalize=False
            )
            
            # Step 5: Extract scalp region (optional)
            self._update_progress(5, "Extracting scalp region...")
            scalp_extraction = self.image_processor.extract_scalp_region(
                enhanced_image, method='adaptive'
            )
            
            # Step 6: Run AI model analysis
            self._update_progress(6, "Running AI model analysis...")
            model_results = self.model_manager.analyze_image(
                enhanced_image,
                save_results=save_results,
                output_dir=output_dir
            )
            
            # Step 7: Process and format results
            self._update_progress(7, "Processing analysis results...")
            processed_results = self.result_processor.process_analysis_results(
                model_results,
                quality_assessment,
                scalp_extraction,
                image_metadata
            )
            
            # Step 8: Generate final report
            self._update_progress(8, "Generating final report...")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Compile final results
            final_results = {
                'success': True,
                'processing_time': processing_time,
                'image_metadata': image_metadata,
                'quality_assessment': quality_assessment,
                'scalp_extraction': scalp_extraction,
                'model_results': model_results,
                'processed_results': processed_results,
                'pipeline_info': {
                    'auto_enhance_used': self.auto_enhance and quality_assessment['overall_quality'] < 80,
                    'quality_threshold': self.quality_threshold,
                    'device_used': self.device,
                    'models_used': model_results['analysis_metadata']['models_used']
                }
            }
            
            # Save results if requested
            if save_results and output_dir:
                self._save_pipeline_results(final_results, output_dir)
            
            logger.info(f"Analysis completed successfully in {processing_time:.2f} seconds")
            return final_results
            
        except Exception as e:
            logger.error(f"Analysis pipeline failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def analyze_batch(self, image_paths: List[str],
                     output_dir: str,
                     save_individual: bool = True) -> Dict:
        """
        Analyze multiple images in batch.
        
        Args:
            image_paths: List of image file paths
            output_dir: Output directory for results
            save_individual: Whether to save individual results
            
        Returns:
            Batch analysis results
        """
        logger.info(f"Starting batch analysis of {len(image_paths)} images")
        
        # Ensure output directory exists
        ensure_dir(output_dir)
        
        batch_results = {
            'total_images': len(image_paths),
            'successful_analyses': 0,
            'failed_analyses': 0,
            'individual_results': [],
            'batch_summary': {},
            'processing_time': 0
        }
        
        start_time = time.time()
        
        for i, image_path in enumerate(image_paths):
            try:
                logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
                
                # Create individual output directory
                individual_output = None
                if save_individual:
                    image_name = Path(image_path).stem
                    individual_output = Path(output_dir) / f"analysis_{image_name}"
                    ensure_dir(individual_output)
                
                # Analyze image
                result = self.analyze_single_image(
                    image_path,
                    output_dir=str(individual_output) if individual_output else None,
                    save_results=save_individual
                )
                
                result['image_path'] = image_path
                result['batch_index'] = i
                
                if result['success']:
                    batch_results['successful_analyses'] += 1
                else:
                    batch_results['failed_analyses'] += 1
                
                batch_results['individual_results'].append(result)
                
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {str(e)}")
                batch_results['failed_analyses'] += 1
                batch_results['individual_results'].append({
                    'image_path': image_path,
                    'batch_index': i,
                    'success': False,
                    'error': str(e)
                })
        
        # Calculate batch processing time
        batch_results['processing_time'] = time.time() - start_time
        
        # Generate batch summary
        batch_results['batch_summary'] = self._generate_batch_summary(batch_results)
        
        # Save batch results
        self._save_batch_results(batch_results, output_dir)
        
        logger.info(f"Batch analysis completed: {batch_results['successful_analyses']}/{batch_results['total_images']} successful")
        
        return batch_results
    
    def _generate_batch_summary(self, batch_results: Dict) -> Dict:
        """Generate summary statistics for batch analysis."""
        successful_results = [
            r for r in batch_results['individual_results'] 
            if r.get('success', False)
        ]
        
        if not successful_results:
            return {'error': 'No successful analyses to summarize'}
        
        # Collect health scores
        health_scores = []
        condition_counts = {}
        severity_counts = {'Mild': 0, 'Moderate': 0, 'Severe': 0}
        
        for result in successful_results:
            if 'processed_results' in result:
                processed = result['processed_results']
                
                # Health scores
                if 'overall_health_score' in processed:
                    health_scores.append(processed['overall_health_score'])
                
                # Condition counts
                if 'detected_conditions' in processed:
                    for condition in processed['detected_conditions']:
                        condition_name = condition.get('condition', 'unknown')
                        condition_counts[condition_name] = condition_counts.get(condition_name, 0) + 1
                        
                        severity = condition.get('severity', 'Unknown')
                        if severity in severity_counts:
                            severity_counts[severity] += 1
        
        summary = {
            'total_successful': len(successful_results),
            'average_health_score': np.mean(health_scores) if health_scores else 0,
            'health_score_std': np.std(health_scores) if health_scores else 0,
            'min_health_score': np.min(health_scores) if health_scores else 0,
            'max_health_score': np.max(health_scores) if health_scores else 0,
            'condition_prevalence': condition_counts,
            'severity_distribution': severity_counts,
            'average_processing_time': np.mean([
                r.get('processing_time', 0) for r in successful_results
            ])
        }
        
        return summary
    
    def _save_pipeline_results(self, results: Dict, output_dir: str):
        """Save complete pipeline results to files."""
        output_path = Path(output_dir)
        ensure_dir(output_path)
        
        # Save JSON results
        import json
        
        # Create serializable version of results
        serializable_results = self._make_serializable(results)
        
        json_path = output_path / "analysis_results.json"
        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Pipeline results saved to: {json_path}")
    
    def _save_batch_results(self, batch_results: Dict, output_dir: str):
        """Save batch analysis results."""
        import json
        
        output_path = Path(output_dir)
        
        # Save batch summary
        serializable_batch = self._make_serializable(batch_results)
        
        json_path = output_path / "batch_analysis_results.json"
        with open(json_path, 'w') as f:
            json.dump(serializable_batch, f, indent=2)
        
        logger.info(f"Batch results saved to: {json_path}")
    
    def _make_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to serializable format."""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return list(self._make_serializable(item) for item in obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            # Handle custom objects by converting to dict
            return self._make_serializable(obj.__dict__)
        else:
            return obj
    
    def get_pipeline_status(self) -> Dict:
        """Get current status of the analysis pipeline."""
        return {
            'models_loaded': self.model_manager.get_model_status(),
            'auto_enhance': self.auto_enhance,
            'quality_threshold': self.quality_threshold,
            'save_intermediate': self.save_intermediate,
            'device': self.device,
            'current_progress': {
                'step': self.current_step,
                'total_steps': self.total_steps
            }
        }
    
    def configure_pipeline(self, **kwargs):
        """
        Configure pipeline settings.
        
        Args:
            auto_enhance: Enable automatic image enhancement
            quality_threshold: Minimum quality score for processing
            save_intermediate: Save intermediate processing results
            analysis_mode: Model analysis mode ('yolo_only', 'cnn_only', 'comprehensive')
        """
        if 'auto_enhance' in kwargs:
            self.auto_enhance = kwargs['auto_enhance']
            logger.info(f"Auto-enhance set to: {self.auto_enhance}")
        
        if 'quality_threshold' in kwargs:
            self.quality_threshold = kwargs['quality_threshold']
            logger.info(f"Quality threshold set to: {self.quality_threshold}")
        
        if 'save_intermediate' in kwargs:
            self.save_intermediate = kwargs['save_intermediate']
            logger.info(f"Save intermediate set to: {self.save_intermediate}")
        
        if 'analysis_mode' in kwargs:
            self.model_manager.set_analysis_mode(kwargs['analysis_mode'])
            logger.info(f"Analysis mode set to: {kwargs['analysis_mode']}")
    
    def validate_setup(self) -> Dict:
        """
        Validate that the pipeline is properly set up.
        
        Returns:
            Validation results with any issues found
        """
        issues = []
        warnings = []
        
        # Check model status
        model_status = self.model_manager.get_model_status()
        
        if not any(model_status['models_loaded'].values()):
            issues.append("No AI models are loaded")
        
        if not model_status['models_loaded']['yolo']:
            warnings.append("YOLO model not loaded - detection capabilities limited")
        
        # Check device availability
        if self.device == 'cuda' and not torch.cuda.is_available():
            warnings.append("CUDA requested but not available, falling back to CPU")
        
        # Check configuration
        if self.quality_threshold < 0 or self.quality_threshold > 100:
            issues.append("Invalid quality threshold - must be between 0 and 100")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'model_status': model_status
        }
    
    def cleanup(self):
        """Clean up pipeline resources."""
        self.model_manager.cleanup()
        logger.info("Pipeline resources cleaned up") 