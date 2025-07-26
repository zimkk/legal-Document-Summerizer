# Legal Document Summarization System - Developer Guide

## Table of Contents
1. [System Overview](#system-overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Dataset Information](#dataset-information)
5. [Development Steps](#development-steps)
6. [Running the System](#running-the-system)
7. [Troubleshooting](#troubleshooting)

## System Overview

This system is designed to process and summarize legal documents using advanced language models. It consists of two main components:
1. Model Training Pipeline (Cloud-based or Local)
2. Local Document Processing System

## Prerequisites

### Software Requirements
- Python 3.8 or higher
- CUDA-compatible GPU (16GB+ VRAM recommended for local training)
- [Ollama](https://ollama.ai/) installed and running
- Git
- Google Colab account (for cloud-based training)

### Python Dependencies
All dependencies are listed in `requirements.txt` with specific version numbers:
```
flask
torch>=2.0.0
transformers>=4.30.0
tokenizers>=0.13.3
accelerate>=0.20.0
bitsandbytes>=0.39.0
protobuf>=3.20.0
numpy>=1.24.0
pandas>=2.0.0
python-docx>=0.8.11
pdf2image>=1.16.3
pytesseract>=0.3.10
sentence-transformers>=2.2.2
requests>=2.31.0
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd legal-ai
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Ollama:
- Visit [Ollama's official website](https://ollama.ai/)
- Download and install the appropriate version for your OS
- Start the Ollama service

## Dataset Information

### Training Dataset
The system uses the BillSum dataset for training, which consists of:
- Training set: 18,949 documents
- Validation set: 3,269 documents
- Test set: 3,269 documents

### Dataset Preparation
The dataset is automatically prepared using `prepare_legal_dataset.py`, which:
1. Downloads the BillSum dataset
2. Processes and cleans the data
3. Splits into training (70%), validation (15%), and test (15%) sets
4. Saves the processed dataset to `datasets/billsum_data.json`

## Development Steps

### Option 1: Cloud-based Training (Recommended)

1. **Google Colab Setup**
   - Create a new Google Colab notebook
   - Mount your Google Drive:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```
   - Upload the following files to your Google Drive:
     - `prepare_legal_dataset.py`
     - `fine_tune_llama3.py`
     - `convert_to_ollama.py`

2. **Install Dependencies in Colab**
   ```python
   !pip install -r requirements.txt
   ```

3. **Run Dataset Preparation**
   ```python
   !python prepare_legal_dataset.py
   ```

4. **Fine-tune the Model**
   ```python
   !python fine_tune_llama3.py
   ```

5. **Convert Model for Ollama**
   ```python
   !python convert_to_ollama.py
   ```

6. **Download the Model**
   - Download the converted model from Google Drive
   - Place it in your local Ollama models directory

### Option 2: Local Training

1. **CUDA Setup**
   - Install NVIDIA drivers
   - Install CUDA Toolkit 11.7 or higher
   - Install cuDNN
   - Verify installation:
     ```bash
     nvidia-smi
     ```

2. **Run Training Pipeline**
   ```bash
   python prepare_legal_dataset.py
   python fine_tune_llama3.py
   python convert_to_ollama.py
   ```

### Local Development Setup

1. **Start Ollama and Load Models**
   ```bash
   # Start the Ollama service
   ollama serve

   # In a new terminal, load the models
   ollama run legal-summarizer
   ollama pull gemma3:4b
   ```

2. **Start the Web Application**
   ```bash
   python app.py
   ```

3. **Access the Web Interface**
   - Open your browser
   - Navigate to `http://localhost:5000`

## Running the System

### Web Application

1. **Start the Flask Server**
   ```bash
   python app.py
   ```

2. **Document Processing**
   - Open `http://localhost:5000` in your browser
   - Upload legal documents (PDF or DOCX format)
   - The system will:
     - Process the document
     - Generate a summary using the legal-summarizer model
     - Verify the summary using the gemma3:4b model
     - Display results with assessment and similarity score

### Command Line Interface

Process documents using the CLI:
```bash
python main.py [path_to_documents]
```

## Troubleshooting

### Common Issues

1. **CUDA Errors**
   - Verify CUDA installation: `nvidia-smi`
   - Check GPU compatibility
   - Ensure sufficient VRAM
   - For Colab users: Ensure GPU runtime is selected

2. **Model Loading Issues**
   - Verify Ollama is running: `ollama list`
   - Check model availability
   - Ensure sufficient disk space
   - Verify model files are in correct location

3. **Document Processing Errors**
   - Verify file formats (PDF/DOCX)
   - Check file permissions
   - Ensure sufficient system memory
   - Check OCR dependencies (Tesseract)

4. **Colab-specific Issues**
   - Runtime disconnection: Save work frequently
   - GPU memory: Monitor usage with `!nvidia-smi`
   - Drive mounting: Re-mount if connection lost

### Getting Help

For additional support:
1. Check the project's issue tracker
2. Review the documentation
3. Contact the development team

## Version Information

- Python: 3.8+
- PyTorch: 2.0.0+
- Transformers: 4.30.0+
- CUDA: 11.7+ (if using GPU)
- Ollama: Latest stable version
- Google Colab: Latest version

## Security Considerations

1. Never commit API keys or sensitive credentials
2. Use environment variables for configuration
3. Validate all user inputs
4. Implement proper error handling
5. Follow security best practices for web applications

## Performance Optimization

1. **Cloud Training**
   - Use Colab Pro for better GPU access
   - Monitor GPU memory usage
   - Save checkpoints frequently
   - Use efficient data loading

2. **Local Processing**
   - Adjust batch sizes based on available VRAM
   - Use multi-threading for document processing
   - Implement caching for frequently accessed data
   - Optimize model loading and inference
   - Monitor system resources during operation

## Additional Resources

1. **Documentation**
   - [Ollama Documentation](https://ollama.ai/docs)
   - [PyTorch Documentation](https://pytorch.org/docs)
   - [Google Colab Guide](https://colab.research.google.com/notebooks/intro.ipynb)

2. **Tutorials**
   - Model fine-tuning guide
   - Document processing workflow
   - Performance optimization tips

3. **Support**
   - GitHub Issues
   - Community Forums
   - Development Team Contact 