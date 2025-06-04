# 🎭 Multimodal Emotion Detection App

A powerful **Streamlit web application** that analyzes emotions from both **audio and video** files using state-of-the-art machine learning models. Perfect for researchers, content creators, and anyone interested in emotion AI!

## 🌟 Features

### 🎵 Audio Emotion Analysis
- **Voice Emotion Detection** using HuBERT (Hidden-Unit BERT) model
- Supports multiple audio formats: WAV, MP3
- Real-time processing with advanced feature extraction
- Detects emotions: Happy, Sad, Angry, Neutral, and more

### 🎥 Video Emotion Analysis  
- **Facial Emotion Recognition** using FER (Facial Emotion Recognition) library
- Supports video formats: MP4, AVI, MOV
- Frame-by-frame analysis with MTCNN face detection
- Analyzes emotions every 2 seconds for efficiency
- Comprehensive emotion breakdown with visual charts

### 🚀 Key Capabilities
- **Multimodal Processing**: Analyze both audio and video simultaneously
- **Real-time Results**: Fast processing with model caching
- **Interactive Interface**: User-friendly Streamlit web app
- **Batch Analysis**: Upload and process multiple files
- **Visual Analytics**: Charts and statistics for emotion distribution

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/multimodal-emotion-detection.git
   cd multimodal-emotion-detection
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

## 📦 Dependencies

Create a `requirements.txt` file with these dependencies:

```
streamlit>=1.28.0
opencv-python>=4.8.0
fer>=22.4.0
pandas>=1.5.0
torch>=2.6.0
torchaudio>=2.1.0
transformers>=4.30.0
mtcnn>=0.1.1
tensorflow>=2.13.0
```

## 🔧 Troubleshooting

### PyTorch Version Error
If you encounter the error: `ValueError: Due to a serious vulnerability issue in torch.load...`

**Solution:**
```bash
pip install --upgrade torch>=2.6.0 torchvision torchaudio
```

### CUDA Support (Optional)
For GPU acceleration:
```bash
pip install torch>=2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Common Issues

| Issue | Solution |
|-------|----------|
| **Model loading fails** | Ensure PyTorch 2.6+ is installed |
| **Video not processing** | Check video codec compatibility |
| **Audio analysis error** | Verify audio file format (WAV/MP3) |
| **Memory issues** | Use smaller video files or reduce frame rate |

## 🎯 Usage Guide

### 1. Launch the App
```bash
streamlit run app.py
```

### 2. Upload Your File
- **Audio Files**: Upload WAV or MP3 files for voice emotion analysis
- **Video Files**: Upload MP4, AVI, or MOV files for facial emotion analysis

### 3. View Results
- **Audio Analysis**: See detected emotion with confidence
- **Video Analysis**: View emotion breakdown across video frames
- **Statistics**: Interactive charts showing emotion distribution

### 4. Interpret Results
- **Dominant Emotion**: Most frequently detected emotion
- **Frame Count**: Number of frames where each emotion was detected
- **Confidence Levels**: Model confidence for predictions

## 🧠 Technical Details

### Models Used

#### Audio Analysis
- **Model**: `superb/hubert-large-superb-er`
- **Architecture**: HuBERT (Hidden-Unit BERT)
- **Input**: 16kHz audio waveforms
- **Output**: Emotion classifications

#### Video Analysis
- **Library**: FER (Facial Emotion Recognition)
- **Face Detection**: MTCNN (Multi-task CNN)
- **Emotions Detected**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

### Processing Pipeline

```
Audio Input → Wav2Vec2 Feature Extraction → HuBERT Model → Emotion Classification
Video Input → Frame Extraction → Face Detection → Emotion Analysis → Aggregation
```

## 📊 Performance

- **Audio Processing**: ~2-5 seconds per minute of audio
- **Video Processing**: ~1-3 seconds per minute of video (depending on resolution)
- **Memory Usage**: ~2-4GB RAM (with model caching)
- **Supported File Sizes**: Up to 100MB per file

## 🔮 Future Enhancements

- [ ] **Real-time webcam emotion detection**
- [ ] **Batch processing for multiple files**
- [ ] **Export results to CSV/JSON**
- [ ] **Advanced emotion analytics dashboard**
- [ ] **Integration with cloud storage**
- [ ] **Mobile app version**
- [ ] **API endpoints for developers**

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Development Setup
```bash
git clone https://github.com/yourusername/multimodal-emotion-detection.git
cd multimodal-emotion-detection
pip install -e .
pre-commit install
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **HuggingFace Transformers** - For the amazing pre-trained models
- **FER Library** - For facial emotion recognition capabilities
- **Streamlit** - For the incredible web app framework
- **PyTorch** - For the deep learning foundation

## 📬 Contact

**Project Maintainer**: [Your Name]
- 📧 Email: your.email@example.com
- 🐦 Twitter: [@yourusername](https://twitter.com/yourusername)
- 💼 LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)

## ⭐ Show Your Support

If this project helped you, please give it a ⭐ on GitHub!

---

## 📈 Project Stats

![GitHub stars](https://img.shields.io/github/stars/yourusername/multimodal-emotion-detection?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/multimodal-emotion-detection?style=social)
![GitHub issues](https://img.shields.io/github/issues/yourusername/multimodal-emotion-detection)

**Made with ❤️ and Python**
