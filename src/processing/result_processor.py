"""
Result Processor for Hair Care AI Application

This module handles post-processing of analysis results, formatting,
and preparation for visualization and reporting.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json
from pathlib import Path

from ..utils.logger import logger
from ..utils.config import config
from ..utils.helpers import format_confidence, generate_recommendations


class ResultProcessor:
    """
    Processes and formats analysis results from AI models.
    
    Handles result aggregation, formatting, statistical analysis,
    and preparation for visualization and reporting.
    """
    
    def __init__(self):
        """Initialize the result processor."""
        self.class_names = config.classes
        
        # Condition severity thresholds
        self.severity_thresholds = {
            'area': {'mild': 5, 'moderate': 15, 'severe': 30},
            'confidence': {'mild': 0.5, 'moderate': 0.7, 'severe': 0.85}
        }
        
        logger.info("ResultProcessor initialized")
    
    def process_analysis_results(self, model_results: Dict,
                                quality_assessment: Dict,
                                scalp_extraction: Dict,
                                image_metadata: Dict) -> Dict:
        """
        Process and format complete analysis results.
        
        Args:
            model_results: Results from model analysis
            quality_assessment: Image quality assessment
            scalp_extraction: Scalp region extraction results
            image_metadata: Original image metadata
            
        Returns:
            Processed and formatted results
        """
        try:
            processed_results = {
                'analysis_timestamp': datetime.now().isoformat(),
                'image_info': self._process_image_info(image_metadata, quality_assessment),
                'detection_summary': self._process_detection_summary(model_results),
                'health_assessment': self._process_health_assessment(model_results),
                'condition_analysis': self._process_condition_analysis(model_results),
                'recommendations': self._process_recommendations(model_results),
                'confidence_metrics': self._calculate_confidence_metrics(model_results),
                'scalp_analysis': self._process_scalp_analysis(scalp_extraction),
                'model_performance': self._assess_model_performance(model_results)
            }
            
            logger.info("Analysis results processed successfully")
            return processed_results
            
        except Exception as e:
            logger.error(f"Result processing failed: {str(e)}")
            raise
    
    def _process_image_info(self, metadata: Dict, quality: Dict) -> Dict:
        """Process image information and quality metrics."""
        return {
            'dimensions': {
                'width': metadata.get('original_size', [0, 0])[0],
                'height': metadata.get('original_size', [0, 0])[1],
                'total_pixels': metadata.get('original_size', [0, 0])[0] * metadata.get('original_size', [0, 0])[1]
            },
            'file_info': {
                'path': metadata.get('file_path'),
                'size_bytes': metadata.get('file_size', 0),
                'format': metadata.get('format', 'Unknown')
            },
            'quality_metrics': {
                'overall_score': quality.get('overall_quality', 0),
                'category': quality.get('quality_category', 'Unknown'),
                'blur_score': quality.get('blur_score', 0),
                'brightness': quality.get('brightness', 0),
                'contrast': quality.get('contrast', 0),
                'saturation': quality.get('saturation', 0),
                'is_suitable': quality.get('overall_quality', 0) >= 60
            }
        }
    
    def _process_detection_summary(self, model_results: Dict) -> Dict:
        """Process detection summary statistics."""
        yolo_results = model_results.get('yolo_results', {})
        
        if not yolo_results or not yolo_results.get('detections'):
            return {
                'total_detections': 0,
                'conditions_detected': [],
                'area_affected': 0.0,
                'detection_confidence': 0.0
            }
        
        detections = yolo_results['detections']
        
        # Count detections by class
        class_counts = {}
        total_area = 0.0
        confidence_scores = []
        
        for detection in detections:
            class_name = detection['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            total_area += detection.get('area_percentage', 0)
            confidence_scores.append(detection.get('confidence', 0))
        
        return {
            'total_detections': len(detections),
            'conditions_detected': list(class_counts.keys()),
            'class_distribution': class_counts,
            'area_affected': min(total_area, 100.0),  # Cap at 100%
            'detection_confidence': np.mean(confidence_scores) if confidence_scores else 0.0,
            'confidence_range': {
                'min': np.min(confidence_scores) if confidence_scores else 0.0,
                'max': np.max(confidence_scores) if confidence_scores else 0.0,
                'std': np.std(confidence_scores) if confidence_scores else 0.0
            }
        }
    
    def _process_health_assessment(self, model_results: Dict) -> Dict:
        """Process overall health assessment."""
        yolo_results = model_results.get('yolo_results', {})
        combined_analysis = model_results.get('combined_analysis', {})
        
        health_score = combined_analysis.get('overall_health_score', 
                                           yolo_results.get('health_score', 0))
        
        # Determine health category
        if health_score >= 85:
            health_category = 'Excellent'
            health_description = 'Your scalp appears very healthy with minimal or no issues detected.'
        elif health_score >= 70:
            health_category = 'Good'
            health_description = 'Your scalp is generally healthy with minor issues that can be easily managed.'
        elif health_score >= 50:
            health_category = 'Fair'
            health_description = 'Your scalp shows some concerns that would benefit from targeted care.'
        elif health_score >= 30:
            health_category = 'Poor'
            health_description = 'Your scalp has significant issues that require attention and care.'
        else:
            health_category = 'Critical'
            health_description = 'Your scalp shows serious conditions that may require professional consultation.'
        
        return {
            'overall_score': health_score,
            'category': health_category,
            'description': health_description,
            'confidence_level': combined_analysis.get('confidence_level', 'Medium'),
            'model_agreement': combined_analysis.get('model_agreement', 'Single Model'),
            'severity_assessment': combined_analysis.get('severity_assessment', 'Unknown')
        }
    
    def _process_condition_analysis(self, model_results: Dict) -> Dict:
        """Process detailed condition analysis."""
        yolo_results = model_results.get('yolo_results', {})
        combined_analysis = model_results.get('combined_analysis', {})
        
        if not yolo_results or not yolo_results.get('detections'):
            return {
                'detected_conditions': [],
                'severity_breakdown': {'Mild': 0, 'Moderate': 0, 'Severe': 0},
                'condition_details': {}
            }
        
        detections = yolo_results['detections']
        condition_details = {}
        severity_counts = {'Mild': 0, 'Moderate': 0, 'Severe': 0}
        
        # Group detections by condition type
        for detection in detections:
            class_name = detection['class_name']
            severity = detection['severity']
            
            if class_name not in condition_details:
                condition_details[class_name] = {
                    'condition_name': self._get_readable_condition_name(class_name),
                    'count': 0,
                    'total_area': 0.0,
                    'avg_confidence': 0.0,
                    'severity_distribution': {'Mild': 0, 'Moderate': 0, 'Severe': 0},
                    'locations': [],
                    'recommendations': []
                }
            
            condition_details[class_name]['count'] += 1
            condition_details[class_name]['total_area'] += detection.get('area_percentage', 0)
            condition_details[class_name]['severity_distribution'][severity] += 1
            condition_details[class_name]['locations'].append({
                'bbox': detection['bbox'],
                'center': detection['center'],
                'confidence': detection['confidence'],
                'severity': severity
            })
            
            severity_counts[severity] += 1
        
        # Calculate averages and generate recommendations
        for class_name, details in condition_details.items():
            if details['count'] > 0:
                # Calculate average confidence
                confidences = [loc['confidence'] for loc in details['locations']]
                details['avg_confidence'] = np.mean(confidences)
                
                # Generate condition-specific recommendations
                details['recommendations'] = generate_recommendations({class_name: details['count']})
        
        return {
            'detected_conditions': list(condition_details.keys()),
            'severity_breakdown': severity_counts,
            'condition_details': condition_details,
            'primary_conditions': combined_analysis.get('primary_conditions', [])
        }
    
    def _process_recommendations(self, model_results: Dict) -> Dict:
        """Process and categorize recommendations."""
        recommendations = model_results.get('recommendations', {})
        
        if not recommendations:
            return {
                'immediate_actions': ['No specific actions required at this time'],
                'care_routine': ['Maintain regular hair care routine'],
                'product_suggestions': ['Use gentle, pH-balanced hair products'],
                'lifestyle_recommendations': ['Maintain healthy diet and hydration'],
                'follow_up_advice': ['Monitor scalp condition regularly'],
                'priority_level': 'Low'
            }
        
        # Determine priority level based on severity
        combined_analysis = model_results.get('combined_analysis', {})
        severity = combined_analysis.get('severity_assessment', 'Unknown')
        
        if severity == 'Severe':
            priority_level = 'High'
        elif severity == 'Moderate':
            priority_level = 'Medium'
        else:
            priority_level = 'Low'
        
        processed_recommendations = recommendations.copy()
        processed_recommendations['priority_level'] = priority_level
        
        return processed_recommendations
    
    def _calculate_confidence_metrics(self, model_results: Dict) -> Dict:
        """Calculate confidence metrics across all models."""
        metrics = {
            'yolo_confidence': 0.0,
            'cnn_confidence': 0.0,
            'ensemble_confidence': 0.0,
            'overall_confidence': 0.0,
            'confidence_distribution': []
        }
        
        # YOLO confidence
        yolo_results = model_results.get('yolo_results', {})
        if yolo_results and yolo_results.get('detections'):
            yolo_confidences = [d['confidence'] for d in yolo_results['detections']]
            metrics['yolo_confidence'] = np.mean(yolo_confidences)
            metrics['confidence_distribution'].extend(yolo_confidences)
        
        # CNN confidence
        scalp_cnn_results = model_results.get('scalp_cnn_results', {})
        if scalp_cnn_results and 'probabilities' in scalp_cnn_results:
            max_prob = np.max(scalp_cnn_results['probabilities'])
            metrics['cnn_confidence'] = max_prob
        
        # Ensemble confidence
        ensemble_results = model_results.get('ensemble_results', {})
        if ensemble_results and 'combined_probabilities' in ensemble_results:
            combined_probs = ensemble_results['combined_probabilities'][0]
            if len(combined_probs) > 0:
                metrics['ensemble_confidence'] = np.max(combined_probs)
        
        # Overall confidence (weighted average)
        confidences = [v for v in [metrics['yolo_confidence'], 
                                 metrics['cnn_confidence'], 
                                 metrics['ensemble_confidence']] if v > 0]
        
        if confidences:
            metrics['overall_confidence'] = np.mean(confidences)
        
        return metrics
    
    def _process_scalp_analysis(self, scalp_extraction: Dict) -> Dict:
        """Process scalp region analysis results."""
        if not scalp_extraction or not scalp_extraction.get('extraction_success', False):
            return {
                'extraction_successful': False,
                'method_used': scalp_extraction.get('method_used', 'unknown'),
                'scalp_coverage': 0.0,
                'analysis_quality': 'Poor'
            }
        
        # Calculate scalp coverage (simplified)
        mask = scalp_extraction.get('mask')
        if mask is not None:
            total_pixels = mask.size
            scalp_pixels = np.sum(mask > 0)
            coverage = (scalp_pixels / total_pixels) * 100
        else:
            coverage = 0.0
        
        # Determine analysis quality
        if coverage > 80:
            analysis_quality = 'Excellent'
        elif coverage > 60:
            analysis_quality = 'Good'
        elif coverage > 40:
            analysis_quality = 'Fair'
        else:
            analysis_quality = 'Poor'
        
        return {
            'extraction_successful': True,
            'method_used': scalp_extraction.get('method_used', 'adaptive'),
            'scalp_coverage': coverage,
            'analysis_quality': analysis_quality
        }
    
    def _assess_model_performance(self, model_results: Dict) -> Dict:
        """Assess the performance of different models used."""
        performance = {
            'models_used': model_results.get('analysis_metadata', {}).get('models_used', []),
            'processing_device': model_results.get('analysis_metadata', {}).get('device', 'unknown'),
            'model_agreement': 'Unknown',
            'reliability_score': 0.0
        }
        
        models_used = performance['models_used']
        
        # Calculate reliability score based on model agreement and confidence
        if len(models_used) > 1:
            combined_analysis = model_results.get('combined_analysis', {})
            agreement = combined_analysis.get('model_agreement', 'Unknown')
            confidence = combined_analysis.get('confidence_level', 'Low')
            
            performance['model_agreement'] = agreement
            
            # Simple reliability scoring
            agreement_scores = {'High': 0.9, 'Medium': 0.7, 'Low': 0.5, 'Unknown': 0.3}
            confidence_scores = {'Very High': 1.0, 'High': 0.8, 'Medium': 0.6, 'Low': 0.4}
            
            reliability = (agreement_scores.get(agreement, 0.3) + 
                         confidence_scores.get(confidence, 0.4)) / 2
            performance['reliability_score'] = reliability
        else:
            performance['reliability_score'] = 0.6  # Single model baseline
        
        return performance
    
    def _get_readable_condition_name(self, class_name: str) -> str:
        """Convert class name to readable condition name."""
        readable_names = {
            'd': 'Dandruff',
            'o': 'Oiliness',
            's': 'Sensitivity',
            'ds': 'Dandruff with Sensitivity',
            'os': 'Oiliness with Sensitivity',
            'dss': 'Multiple Sensitivities with Dandruff'
        }
        return readable_names.get(class_name, class_name)
    
    def generate_summary_report(self, processed_results: Dict) -> str:
        """Generate a human-readable summary report."""
        try:
            health_assessment = processed_results['health_assessment']
            detection_summary = processed_results['detection_summary']
            condition_analysis = processed_results['condition_analysis']
            
            report_lines = []
            
            # Header
            report_lines.append("=== SCALP HEALTH ANALYSIS REPORT ===")
            report_lines.append(f"Analysis Date: {processed_results['analysis_timestamp']}")
            report_lines.append("")
            
            # Overall Health Assessment
            report_lines.append("OVERALL HEALTH ASSESSMENT:")
            report_lines.append(f"Health Score: {health_assessment['overall_score']:.1f}/100")
            report_lines.append(f"Category: {health_assessment['category']}")
            report_lines.append(f"Description: {health_assessment['description']}")
            report_lines.append("")
            
            # Detection Summary
            if detection_summary['total_detections'] > 0:
                report_lines.append("DETECTED CONDITIONS:")
                for condition in detection_summary['conditions_detected']:
                    readable_name = self._get_readable_condition_name(condition)
                    count = detection_summary['class_distribution'][condition]
                    report_lines.append(f"- {readable_name}: {count} detection(s)")
                
                report_lines.append(f"Total Area Affected: {detection_summary['area_affected']:.1f}%")
                report_lines.append("")
            else:
                report_lines.append("No scalp conditions detected.")
                report_lines.append("")
            
            # Recommendations
            recommendations = processed_results['recommendations']
            if recommendations.get('immediate_actions'):
                report_lines.append("IMMEDIATE ACTIONS:")
                for action in recommendations['immediate_actions'][:3]:  # Top 3
                    report_lines.append(f"- {action}")
                report_lines.append("")
            
            if recommendations.get('care_routine'):
                report_lines.append("CARE ROUTINE RECOMMENDATIONS:")
                for routine in recommendations['care_routine'][:3]:  # Top 3
                    report_lines.append(f"- {routine}")
                report_lines.append("")
            
            # Confidence and Reliability
            confidence_metrics = processed_results['confidence_metrics']
            model_performance = processed_results['model_performance']
            
            report_lines.append("ANALYSIS CONFIDENCE:")
            report_lines.append(f"Overall Confidence: {confidence_metrics['overall_confidence']:.2f}")
            report_lines.append(f"Model Agreement: {model_performance['model_agreement']}")
            report_lines.append(f"Reliability Score: {model_performance['reliability_score']:.2f}")
            
            return "\n".join(report_lines)
            
        except Exception as e:
            logger.error(f"Failed to generate summary report: {str(e)}")
            return "Error generating summary report"
    
    def export_results_json(self, processed_results: Dict, output_path: str) -> bool:
        """Export processed results to JSON file."""
        try:
            # Make results JSON serializable
            serializable_results = self._make_json_serializable(processed_results)
            
            with open(output_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Results exported to JSON: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export JSON results: {str(e)}")
            return False
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj
    
    def compare_results(self, results1: Dict, results2: Dict) -> Dict:
        """Compare two analysis results for progress tracking."""
        try:
            comparison = {
                'health_score_change': 0.0,
                'condition_changes': {},
                'improvement_detected': False,
                'recommendations': []
            }
            
            # Compare health scores
            score1 = results1.get('health_assessment', {}).get('overall_score', 0)
            score2 = results2.get('health_assessment', {}).get('overall_score', 0)
            
            comparison['health_score_change'] = score2 - score1
            comparison['improvement_detected'] = comparison['health_score_change'] > 0
            
            # Compare detected conditions
            conditions1 = set(results1.get('condition_analysis', {}).get('detected_conditions', []))
            conditions2 = set(results2.get('condition_analysis', {}).get('detected_conditions', []))
            
            comparison['condition_changes'] = {
                'new_conditions': list(conditions2 - conditions1),
                'resolved_conditions': list(conditions1 - conditions2),
                'persistent_conditions': list(conditions1 & conditions2)
            }
            
            # Generate comparison recommendations
            if comparison['improvement_detected']:
                comparison['recommendations'].append("Great progress! Continue current care routine.")
            elif comparison['health_score_change'] < -5:
                comparison['recommendations'].append("Condition may be worsening. Consider professional consultation.")
            else:
                comparison['recommendations'].append("Condition appears stable. Maintain current care routine.")
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare results: {str(e)}")
            return {'error': str(e)} 