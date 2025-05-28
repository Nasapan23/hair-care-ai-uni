# Hair Care AI - Professional Scalp Health Analysis

A comprehensive Python-based application that combines YOLOv11 with additional CNN architectures to provide professional-grade scalp health analysis and personalized care recommendations.

![Hair Care AI](https://img.shields.io/badge/Hair%20Care%20AI-v1.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🌟 Features

### 🔬 Advanced AI Analysis
- **YOLOv11 Object Detection**: Precise localization of scalp conditions
- **Custom CNN Architectures**: ScalpCNN and HairHealthCNN for enhanced analysis
- **Ensemble Learning**: Combines multiple models for robust predictions
- **Real-time Processing**: Analysis completed in under 3 seconds

### 🎯 Scalp Condition Detection
- **Dandruff Detection**: Identifies dandruff particles and affected areas
- **Oiliness Assessment**: Analyzes scalp oil levels and distribution
- **Sensitivity Analysis**: Detects irritation and inflammatory conditions
- **Combined Conditions**: Handles multiple simultaneous conditions

### 📊 Comprehensive Health Metrics
- **Health Score**: 0-100 scale overall scalp health rating
- **Severity Assessment**: Mild, Moderate, Severe classification
- **Area Analysis**: Percentage of scalp affected by conditions
- **Confidence Ratings**: Model certainty and agreement metrics

### 💡 Personalized Recommendations
- **Immediate Actions**: Urgent care recommendations
- **Care Routines**: Daily and weekly scalp care suggestions
- **Product Recommendations**: Targeted product suggestions
- **Lifestyle Advice**: Holistic health recommendations
- **Follow-up Guidance**: Progress tracking suggestions

### 🖥️ Multiple Interfaces
- **Modern Web GUI**: Streamlit-based professional interface
- **Command Line**: Batch processing and automation support
- **Batch Processing**: Analyze multiple images simultaneously
- **API Ready**: Modular design for easy integration

## 📋 Requirements

### System Requirements
- **Operating System**: Windows 10+, macOS 10.14+, Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB free space
- **GPU**: CUDA-compatible GPU recommended (optional)

### Image Requirements
- **Formats**: JPG, JPEG, PNG, BMP
- **Resolution**: 224x224 to 4096x4096 pixels
- **File Size**: Under 50MB
- **Quality**: Clear, well-lit images for best results

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/hair-care-ai-uni.git
cd hair-care-ai-uni
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python main.py --validate
```

## 🎮 Usage

### GUI Application (Recommended)
Launch the modern web interface:
```bash
python main.py --gui
```

Then open your browser to the displayed URL (typically `http://localhost:8501`)

### Command Line Interface
Analyze a single image:
```bash
python main.py --cli --image path/to/scalp_image.jpg --output results/
```

### Batch Processing
Process multiple images:
```bash
python main.py --batch --input images_folder/ --output results/
```

### Advanced Options
```bash
# Use only YOLO model
python main.py --cli --image image.jpg --mode yolo_only

# Adjust confidence threshold
python main.py --cli --image image.jpg --confidence 0.3

# Disable auto-enhancement
python main.py --cli --image image.jpg --no-enhance

# Set quality threshold
python main.py --cli --image image.jpg --quality-threshold 70
```

## 📁 Project Structure

```
hair-care-ai-uni/
├── src/                          # Source code
│   ├── __init__.py
│   ├── models/                   # AI models
│   │   ├── yolo_model.py        # YOLOv11 wrapper
│   │   ├── cnn_models.py        # Custom CNN architectures
│   │   └── model_manager.py     # Model coordination
│   ├── processing/              # Image processing
│   │   ├── image_processor.py   # Image preprocessing
│   │   ├── analysis_pipeline.py # Analysis workflow
│   │   └── result_processor.py  # Result formatting
│   ├── gui/                     # User interfaces
│   │   ├── streamlit_app.py     # Web GUI
│   │   └── main_app.py          # Application entry
│   └── utils/                   # Utilities
│       ├── config.py            # Configuration
│       ├── logger.py            # Logging
│       └── helpers.py           # Helper functions
├── data/                        # Data directory
│   ├── sample_images/           # Example images
│   └── results/                 # Analysis outputs
├── tests/                       # Test suite
├── docs/                        # Documentation
├── best.pt                      # Trained YOLOv11 model
├── config.yaml                  # Configuration file
├── requirements.txt             # Dependencies
├── main.py                      # Main entry point
└── README.md                    # This file
```

## 🔧 Configuration

### Model Settings
Edit `config.yaml` to customize:
```yaml
models:
  confidence_threshold: 0.25      # Detection confidence
  image_size: 640                 # Input image size
  
app:
  max_file_size: 50              # Max file size (MB)
  supported_formats: ["jpg", "jpeg", "png", "bmp"]
```

### Analysis Modes
- **comprehensive**: Uses all available models (recommended)
- **yolo_only**: YOLOv11 detection only (fastest)
- **cnn_only**: CNN models only (experimental)

## 📊 Model Performance

### YOLOv11 Metrics
- **Overall mAP50**: 0.253
- **Overall mAP50-95**: 0.12
- **Processing Speed**: <3 seconds per image
- **Model Size**: 18MB

### Individual Class Performance
| Condition | mAP50 | mAP50-95 | Description |
|-----------|-------|----------|-------------|
| Dandruff (d) | 0.536 | 0.309 | Best performing class |
| Oiliness (o) | 0.158 | 0.072 | Moderate performance |
| Sensitivity (s) | 0.275 | 0.074 | Good detection |
| Combined (ds, os, dss) | 0.042 | 0.025 | Challenging cases |

## 🎨 GUI Features

### Modern Interface
- **Responsive Design**: Works on desktop and tablet
- **Real-time Progress**: Live analysis progress tracking
- **Interactive Charts**: Plotly-powered visualizations
- **Export Options**: Download reports and data

### Analysis Tabs
1. **Detection Results**: Visual condition mapping
2. **Health Metrics**: Detailed scoring and confidence
3. **Recommendations**: Personalized care advice
4. **Detailed Report**: Comprehensive analysis summary

### History Tracking
- **Session History**: Track multiple analyses
- **Trend Analysis**: Health score progression
- **Comparison Tools**: Before/after analysis

## 🔬 Technical Details

### AI Architecture
- **YOLOv11s**: 238 layers, 9.4M parameters
- **ScalpCNN**: VGG-inspired architecture with batch normalization
- **HairHealthCNN**: Lightweight depthwise separable convolutions
- **Ensemble**: Weighted fusion with attention mechanism

### Image Processing Pipeline
1. **Quality Assessment**: Blur, brightness, contrast analysis
2. **Enhancement**: Automatic quality improvement
3. **Preprocessing**: Resizing, normalization, padding
4. **Scalp Extraction**: Region-of-interest detection
5. **Multi-model Inference**: Parallel model execution
6. **Result Fusion**: Ensemble decision making

### Performance Optimization
- **Device Auto-detection**: CUDA/MPS/CPU optimization
- **Memory Management**: Efficient resource utilization
- **Batch Processing**: Parallel image processing
- **Model Caching**: Fast subsequent analyses

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/ -v
```

Test specific components:
```bash
# Test models
python -m pytest tests/test_models.py

# Test processing
python -m pytest tests/test_processing.py

# Test GUI
python -m pytest tests/test_gui.py
```

## 📈 Results and Reports

### Analysis Output
Each analysis generates:
- **Annotated Images**: Visual detection overlays
- **JSON Data**: Structured analysis results
- **Text Reports**: Human-readable summaries
- **Confidence Metrics**: Model reliability scores

### Report Sections
1. **Executive Summary**: Key findings and health score
2. **Detailed Analysis**: Condition-by-condition breakdown
3. **Recommendations**: Categorized care suggestions
4. **Technical Details**: Model performance and confidence

## 🔍 Troubleshooting

### Common Issues

**Model Loading Errors**
```bash
# Verify model file exists
ls -la best.pt

# Check file permissions
chmod 644 best.pt
```

**Memory Issues**
```bash
# Use CPU mode
python main.py --cli --image image.jpg --device cpu

# Reduce batch size
python main.py --batch --input images/ --output results/ --batch-size 8
```

**GUI Not Loading**
```bash
# Install Streamlit
pip install streamlit

# Check port availability
netstat -an | grep 8501
```

### Performance Tips
- Use GPU acceleration when available
- Ensure good image quality for best results
- Close other applications to free memory
- Use batch processing for multiple images

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run pre-commit hooks
pre-commit install

# Run tests
python -m pytest
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: [Scalp Data from Roboflow](https://universe.roboflow.com/buyumedatasets2/scalp-data)
- **YOLOv11**: Ultralytics team for the excellent framework
- **PyTorch**: Facebook AI Research team
- **Streamlit**: Streamlit team for the amazing GUI framework

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/hair-care-ai-uni/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/hair-care-ai-uni/discussions)
- **Email**: your-email@example.com

## 🗺️ Roadmap

### Version 1.1 (Planned)
- [ ] Mobile app interface
- [ ] Cloud deployment options
- [ ] Advanced visualization tools
- [ ] Multi-language support

### Version 1.2 (Future)
- [ ] Video analysis capabilities
- [ ] Treatment tracking
- [ ] Professional dashboard
- [ ] API documentation

---

**Hair Care AI** - Empowering better scalp health through artificial intelligence 🧴✨