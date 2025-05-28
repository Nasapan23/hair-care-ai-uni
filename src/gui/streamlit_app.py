"""
Streamlit GUI Application for Hair Care AI

This module provides a modern, web-based interface for the scalp health
analysis system using Streamlit.
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from pathlib import Path
import io
import base64

from ..processing.analysis_pipeline import AnalysisPipeline
from ..utils.logger import logger
from ..utils.config import config


class StreamlitApp:
    """Main Streamlit application for Hair Care AI."""
    
    def __init__(self):
        """Initialize the Streamlit application."""
        self.setup_page_config()
        self.initialize_session_state()
    
    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Hair Care AI - Scalp Health Analysis",
            page_icon="🧴",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better styling
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #2E86AB;
            text-align: center;
            margin-bottom: 2rem;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #A23B72;
            margin-bottom: 1rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #2E86AB;
        }
        .health-excellent { border-left-color: #28a745; }
        .health-good { border-left-color: #17a2b8; }
        .health-fair { border-left-color: #ffc107; }
        .health-poor { border-left-color: #fd7e14; }
        .health-critical { border-left-color: #dc3545; }
        .recommendation-box {
            background-color: #e8f4f8;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'pipeline_initialized' not in st.session_state:
            st.session_state.pipeline_initialized = False
        
        if 'pipeline' not in st.session_state:
            st.session_state.pipeline = None
        
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        
        if 'uploaded_image' not in st.session_state:
            st.session_state.uploaded_image = None
        
        if 'analysis_history' not in st.session_state:
            st.session_state.analysis_history = []
    
    def run(self):
        """Run the main Streamlit application."""
        # Header
        st.markdown('<h1 class="main-header">🧴 Hair Care AI</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Professional Scalp Health Analysis using Computer Vision</p>', unsafe_allow_html=True)
        
        # Sidebar
        self.render_sidebar()
        
        # Main content
        if not st.session_state.pipeline_initialized:
            self.render_setup_page()
        else:
            self.render_main_interface()
    
    def render_sidebar(self):
        """Render the sidebar with navigation and settings."""
        with st.sidebar:
            st.markdown("## 🔧 Settings")
            
            # Model configuration
            st.markdown("### Model Configuration")
            
            analysis_mode = st.selectbox(
                "Analysis Mode",
                ["comprehensive", "yolo_only", "cnn_only"],
                index=0,
                help="Choose which AI models to use for analysis"
            )
            
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.1,
                max_value=0.9,
                value=0.25,
                step=0.05,
                help="Minimum confidence for detections"
            )
            
            auto_enhance = st.checkbox(
                "Auto-enhance Images",
                value=True,
                help="Automatically enhance image quality for better analysis"
            )
            
            # Pipeline initialization
            st.markdown("### System Status")
            
            if st.button("🚀 Initialize AI Models", type="primary"):
                self.initialize_pipeline(analysis_mode, confidence_threshold, auto_enhance)
            
            if st.session_state.pipeline_initialized:
                st.success("✅ AI Models Ready")
                
                # Show model status
                if st.session_state.pipeline is not None:
                    status = st.session_state.pipeline.get_pipeline_status()
                    models_loaded = status['models_loaded']['models_loaded']
                    
                    st.markdown("**Loaded Models:**")
                    # Only show models that are actually loaded
                    loaded_models = {k: v for k, v in models_loaded.items() if v}
                    
                    if loaded_models:
                        for model, loaded in loaded_models.items():
                            st.write(f"✅ {model.upper()}")
                    else:
                        st.write("No models loaded yet")
                    
                    # Show analysis capabilities based on loaded models
                    if models_loaded.get('yolo', False):
                        st.success("🎯 Object Detection Ready")
                    if any([models_loaded.get('scalp_cnn', False), 
                           models_loaded.get('hair_cnn', False)]):
                        st.success("🧠 CNN Analysis Ready")
                        st.info("ℹ️ CNN models use random weights for demonstration")
                    if models_loaded.get('ensemble', False):
                        st.success("🔗 Ensemble Analysis Ready")
            else:
                st.warning("⚠️ Please initialize AI models first")
            
            # About section
            st.markdown("---")
            st.markdown("### About")
            st.info(
                "Hair Care AI uses advanced computer vision to analyze scalp health. "
                "Upload an image of your scalp to get detailed analysis and personalized recommendations."
            )
    
    def initialize_pipeline(self, analysis_mode, confidence_threshold, auto_enhance):
        """Initialize the analysis pipeline."""
        try:
            with st.spinner("Initializing AI models... This may take a moment."):
                pipeline = AnalysisPipeline()
                
                # Configure pipeline
                pipeline.configure_pipeline(
                    analysis_mode=analysis_mode,
                    auto_enhance=auto_enhance
                )
                
                # Initialize models
                model_status = pipeline.initialize_models()
                
                if any(model_status.values()):
                    st.session_state.pipeline = pipeline
                    st.session_state.pipeline_initialized = True
                    st.success("AI models initialized successfully!")
                    logger.info("Pipeline initialized via Streamlit interface")
                else:
                    st.error("Failed to initialize AI models. Please check the model files.")
                    
        except Exception as e:
            st.error(f"Error initializing pipeline: {str(e)}")
            logger.error(f"Pipeline initialization failed: {str(e)}")
    
    def render_setup_page(self):
        """Render the initial setup page."""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("## 🚀 Getting Started")
            
            st.markdown("""
            Welcome to Hair Care AI! Follow these steps to get started:
            
            1. **Initialize AI Models** - Click the button in the sidebar to load the AI models
            2. **Upload Image** - Upload a clear photo of your scalp
            3. **Get Analysis** - Receive detailed health assessment and recommendations
            
            ### 📋 Requirements for Best Results:
            - Clear, well-lit image of your scalp
            - Image size: 224x224 to 4096x4096 pixels
            - Supported formats: JPG, PNG, BMP
            - File size: Under 50MB
            """)
            
            st.markdown("### 🔬 What We Analyze:")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("""
                **Scalp Conditions:**
                - 🔴 Dandruff detection
                - 🟢 Oiliness assessment
                - 🔵 Sensitivity analysis
                - 🟡 Combined conditions
                """)
            
            with col_b:
                st.markdown("""
                **Health Metrics:**
                - Overall health score
                - Severity assessment
                - Area affected analysis
                - Confidence ratings
                """)
    
    def render_main_interface(self):
        """Render the main application interface."""
        # Image upload section
        st.markdown('<h2 class="sub-header">📸 Upload Scalp Image</h2>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of your scalp for analysis"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.session_state.uploaded_image = image
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)
            
            with col2:
                st.markdown("**Image Information:**")
                st.write(f"📏 Size: {image.size[0]} x {image.size[1]} pixels")
                st.write(f"🎨 Mode: {image.mode}")
                st.write(f"📁 Format: {uploaded_file.type}")
                
                # Analysis options
                st.markdown("**Analysis Options:**")
                save_results = st.checkbox("Save Results", value=True)
                
                # Analysis button
                if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                    if st.session_state.pipeline is not None:
                        self.analyze_image(image, save_results)
                    else:
                        st.error("❌ Please initialize the AI models first using the sidebar.")
        
        # Show analysis results if available
        if st.session_state.analysis_results:
            # Show image comparison if we have annotated image
            if ('model_results' in st.session_state.analysis_results and 
                'annotated_image' in st.session_state.analysis_results['model_results']):
                
                st.markdown('<h2 class="sub-header">📸 Image Comparison</h2>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Original Image**")
                    if st.session_state.uploaded_image:
                        st.image(st.session_state.uploaded_image, use_container_width=True)
                
                with col2:
                    st.markdown("**Detected Conditions**")
                    annotated_image = st.session_state.analysis_results['model_results']['annotated_image']
                    st.image(annotated_image, use_container_width=True)
                
                # Quick stats below images
                yolo_results = st.session_state.analysis_results['model_results'].get('yolo_results', {})
                if yolo_results:
                    col_a, col_b, col_c, col_d = st.columns(4)
                    
                    with col_a:
                        st.metric("Health Score", f"{yolo_results.get('health_score', 0)}/100")
                    
                    with col_b:
                        st.metric("Detections", len(yolo_results.get('detections', [])))
                    
                    with col_c:
                        st.metric("Area Affected", f"{yolo_results.get('total_area_affected', 0):.1f}%")
                    
                    with col_d:
                        processing_time = st.session_state.analysis_results.get('processing_time', 0)
                        st.metric("Processing Time", f"{processing_time:.2f}s")
                
                st.markdown("---")
            
            self.render_analysis_results()
        
        # Analysis history
        if st.session_state.analysis_history:
            with st.expander("📚 Analysis History"):
                self.render_analysis_history()
    
    def analyze_image(self, image, save_results):
        """Analyze the uploaded image."""
        try:
            # Check if pipeline is initialized
            if st.session_state.pipeline is None:
                st.error("❌ Pipeline not initialized. Please initialize AI models first.")
                return
            
            # Convert PIL image to numpy array
            image_array = np.array(image)
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def progress_callback(current, total, description):
                progress = current / total
                progress_bar.progress(progress)
                status_text.text(f"Step {current}/{total}: {description}")
            
            # Set progress callback
            st.session_state.pipeline.set_progress_callback(progress_callback)
            
            # Run analysis
            with st.spinner("Analyzing image..."):
                results = st.session_state.pipeline.analyze_single_image(
                    image_array,
                    output_dir="data/results" if save_results else None,
                    save_results=save_results
                )
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            if results['success']:
                st.session_state.analysis_results = results
                st.session_state.analysis_history.append({
                    'timestamp': time.time(),
                    'results': results,
                    'image_size': image.size
                })
                st.success("✅ Analysis completed successfully!")
                st.rerun()
            else:
                st.error(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")
                
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            logger.error(f"Streamlit analysis failed: {str(e)}")
    
    def render_analysis_results(self):
        """Render the analysis results."""
        results = st.session_state.analysis_results
        processed_results = results['processed_results']
        
        st.markdown('<h2 class="sub-header">📊 Analysis Results</h2>', unsafe_allow_html=True)
        
        # Health score overview
        self.render_health_overview(processed_results)
        
        # Detailed analysis tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Detection Results", "📈 Health Metrics", "💡 Recommendations", "📋 Detailed Report"])
        
        with tab1:
            self.render_detection_results(processed_results)
        
        with tab2:
            self.render_health_metrics(processed_results)
        
        with tab3:
            self.render_recommendations(processed_results)
        
        with tab4:
            self.render_detailed_report(processed_results)
    
    def render_health_overview(self, processed_results):
        """Render the health score overview."""
        health_assessment = processed_results['health_assessment']
        detection_summary = processed_results['detection_summary']
        
        # Health score card
        score = health_assessment['overall_score']
        category = health_assessment['category']
        
        # Determine color based on category
        color_map = {
            'Excellent': '#28a745',
            'Good': '#17a2b8', 
            'Fair': '#ffc107',
            'Poor': '#fd7e14',
            'Critical': '#dc3545'
        }
        color = color_map.get(category, '#6c757d')
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Health Score",
                value=f"{score:.1f}/100",
                delta=None
            )
        
        with col2:
            st.metric(
                label="Health Category",
                value=category,
                delta=None
            )
        
        with col3:
            st.metric(
                label="Conditions Detected",
                value=detection_summary['total_detections'],
                delta=None
            )
        
        with col4:
            st.metric(
                label="Area Affected",
                value=f"{detection_summary['area_affected']:.1f}%",
                delta=None
            )
        
        # Health description
        st.markdown(f"""
        <div class="recommendation-box">
        <strong>Assessment:</strong> {health_assessment['description']}
        </div>
        """, unsafe_allow_html=True)
    
    def render_detection_results(self, processed_results):
        """Render detection results visualization."""
        detection_summary = processed_results['detection_summary']
        condition_analysis = processed_results['condition_analysis']
        
        # Show annotated image if available
        if 'annotated_image' in st.session_state.analysis_results['model_results']:
            st.markdown("### 🎯 Detection Visualization")
            annotated_image = st.session_state.analysis_results['model_results']['annotated_image']
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(
                    annotated_image, 
                    caption="Detected Conditions with Bounding Boxes", 
                    use_container_width=True
                )
            
            with col2:
                st.markdown("**Legend:**")
                # Show color legend for different conditions
                yolo_results = st.session_state.analysis_results['model_results'].get('yolo_results', {})
                if yolo_results and 'detections' in yolo_results:
                    unique_conditions = set()
                    for detection in yolo_results['detections']:
                        unique_conditions.add(detection['class_name'])
                    
                    condition_names = {
                        'd': '🔴 Dandruff',
                        'o': '🟢 Oiliness',
                        's': '🔵 Sensitivity',
                        'ds': '🟡 Dandruff + Sensitivity',
                        'os': '🟠 Oiliness + Sensitivity',
                        'dss': '🟣 Multiple Conditions'
                    }
                    
                    for condition in unique_conditions:
                        if condition in condition_names:
                            st.write(condition_names[condition])
                
                # Show detection statistics
                if yolo_results and 'detections' in yolo_results:
                    st.markdown("**Detection Stats:**")
                    st.write(f"📊 Total Detections: {len(yolo_results['detections'])}")
                    st.write(f"🎯 Health Score: {yolo_results.get('health_score', 'N/A')}/100")
                    st.write(f"📏 Area Affected: {yolo_results.get('total_area_affected', 0):.1f}%")
            
            st.markdown("---")
        
        if detection_summary['total_detections'] == 0:
            st.info("🎉 No scalp conditions detected! Your scalp appears healthy.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Condition distribution pie chart
            if detection_summary['class_distribution']:
                fig = px.pie(
                    values=list(detection_summary['class_distribution'].values()),
                    names=[self._get_readable_name(name) for name in detection_summary['class_distribution'].keys()],
                    title="Detected Conditions Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Severity breakdown
            severity_data = condition_analysis['severity_breakdown']
            if any(severity_data.values()):
                fig = px.bar(
                    x=list(severity_data.keys()),
                    y=list(severity_data.values()),
                    title="Severity Breakdown",
                    color=list(severity_data.keys()),
                    color_discrete_map={
                        'Mild': '#28a745',
                        'Moderate': '#ffc107', 
                        'Severe': '#dc3545'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Detailed condition information
        st.markdown("### Condition Details")
        
        for condition, details in condition_analysis['condition_details'].items():
            with st.expander(f"{details['condition_name']} ({details['count']} detection(s))"):
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("Count", details['count'])
                
                with col_b:
                    st.metric("Total Area", f"{details['total_area']:.1f}%")
                
                with col_c:
                    st.metric("Avg Confidence", f"{details['avg_confidence']:.2f}")
                
                # Severity distribution for this condition
                severity_dist = details['severity_distribution']
                if any(severity_dist.values()):
                    fig = px.bar(
                        x=list(severity_dist.keys()),
                        y=list(severity_dist.values()),
                        title=f"Severity Distribution - {details['condition_name']}",
                        color=list(severity_dist.keys()),
                        color_discrete_map={
                            'Mild': '#28a745',
                            'Moderate': '#ffc107',
                            'Severe': '#dc3545'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    def render_health_metrics(self, processed_results):
        """Render detailed health metrics."""
        confidence_metrics = processed_results['confidence_metrics']
        model_performance = processed_results['model_performance']
        image_info = processed_results['image_info']
        
        # Confidence metrics
        st.markdown("### 🎯 Confidence Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Overall Confidence", f"{confidence_metrics['overall_confidence']:.2f}")
        
        with col2:
            st.metric("Model Agreement", model_performance['model_agreement'])
        
        with col3:
            st.metric("Reliability Score", f"{model_performance['reliability_score']:.2f}")
        
        # Image quality metrics
        st.markdown("### 📸 Image Quality Assessment")
        
        quality_metrics = image_info['quality_metrics']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Quality Score", f"{quality_metrics['overall_score']:.1f}/100")
        
        with col2:
            st.metric("Category", quality_metrics['category'])
        
        with col3:
            st.metric("Brightness", f"{quality_metrics['brightness']:.1f}")
        
        with col4:
            st.metric("Contrast", f"{quality_metrics['contrast']:.1f}")
        
        # Processing information
        st.markdown("### ⚙️ Processing Information")
        
        processing_time = st.session_state.analysis_results['processing_time']
        models_used = model_performance['models_used']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Processing Time", f"{processing_time:.2f}s")
        
        with col2:
            st.write("**Models Used:**")
            for model in models_used:
                st.write(f"• {model.upper()}")
    
    def render_recommendations(self, processed_results):
        """Render care recommendations."""
        recommendations = processed_results['recommendations']
        
        # Priority level
        priority = recommendations.get('priority_level', 'Medium')
        priority_colors = {
            'High': '#dc3545',
            'Medium': '#ffc107',
            'Low': '#28a745'
        }
        
        st.markdown(f"""
        <div style="background-color: {priority_colors.get(priority, '#6c757d')}20; 
                    padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
        <h4 style="color: {priority_colors.get(priority, '#6c757d')};">
        Priority Level: {priority}
        </h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Recommendation categories
        categories = [
            ('immediate_actions', '🚨 Immediate Actions', '#dc3545'),
            ('care_routine', '🧴 Care Routine', '#17a2b8'),
            ('product_suggestions', '🛒 Product Suggestions', '#28a745'),
            ('lifestyle_recommendations', '🌱 Lifestyle', '#6f42c1'),
            ('follow_up_advice', '📅 Follow-up', '#fd7e14')
        ]
        
        for key, title, color in categories:
            if recommendations.get(key):
                st.markdown(f"### {title}")
                
                for recommendation in recommendations[key]:
                    st.markdown(f"""
                    <div class="recommendation-box" style="border-left: 4px solid {color};">
                    • {recommendation}
                    </div>
                    """, unsafe_allow_html=True)
    
    def render_detailed_report(self, processed_results):
        """Render detailed analysis report."""
        # Generate text report
        from ..processing.result_processor import ResultProcessor
        processor = ResultProcessor()
        
        report_text = processor.generate_summary_report(processed_results)
        
        st.markdown("### 📋 Complete Analysis Report")
        
        # Display report in a text area
        st.text_area(
            "Detailed Report",
            value=report_text,
            height=400,
            help="Complete analysis report with all findings and recommendations"
        )
        
        # Download options
        col1, col2 = st.columns(2)
        
        with col1:
            # Download as text
            st.download_button(
                label="📄 Download Report (TXT)",
                data=report_text,
                file_name=f"scalp_analysis_report_{int(time.time())}.txt",
                mime="text/plain"
            )
        
        with col2:
            # Download as JSON
            import json
            json_data = json.dumps(processed_results, indent=2, default=str)
            st.download_button(
                label="📊 Download Data (JSON)",
                data=json_data,
                file_name=f"scalp_analysis_data_{int(time.time())}.json",
                mime="application/json"
            )
    
    def render_analysis_history(self):
        """Render analysis history."""
        st.markdown('<h2 class="sub-header">📚 Analysis History</h2>', unsafe_allow_html=True)
        
        history = st.session_state.analysis_history
        
        if len(history) > 1:
            # Show trend analysis
            scores = [h['results']['processed_results']['health_assessment']['overall_score'] 
                     for h in history if h['results']['success']]
            
            if len(scores) > 1:
                fig = px.line(
                    x=list(range(1, len(scores) + 1)),
                    y=scores,
                    title="Health Score Trend",
                    labels={'x': 'Analysis Session', 'y': 'Health Score'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # History table
        if history:
            history_data = []
            for i, h in enumerate(history):
                if h['results']['success']:
                    processed = h['results']['processed_results']
                    history_data.append({
                        'Session': i + 1,
                        'Date': pd.to_datetime(h['timestamp'], unit='s').strftime('%Y-%m-%d %H:%M'),
                        'Health Score': f"{processed['health_assessment']['overall_score']:.1f}",
                        'Category': processed['health_assessment']['category'],
                        'Conditions': processed['detection_summary']['total_detections'],
                        'Processing Time': f"{h['results']['processing_time']:.2f}s"
                    })
            
            if history_data:
                df = pd.DataFrame(history_data)
                st.dataframe(df, use_container_width=True)
    
    def _get_readable_name(self, class_name):
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


def run_streamlit_app():
    """Run the Streamlit application."""
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    run_streamlit_app() 