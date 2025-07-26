# Legal Document Summarization System

## Workflow Overview

**1. Model Preparation (Cloud/Colab)**
- Run `prepare_legal_dataset.py`, `fine_tune_llama3.py`, and `convert_to_ollama.py` in a cloud environment (e.g., Google Colab) to prepare and convert your custom model for Ollama.
- These steps require significant compute resources and are not intended for local execution.

**2. Local Document Processing**
- Use the web application (`app.py`) and supporting scripts (`main.py`, `gemma_analyzer.py`) to process, summarize, and verify legal documents locally using your custom Ollama models.
- The web interface (`templates/index.html`) allows you to upload documents, generate summaries, and check summary accuracy using your custom Ollama models.

---

A comprehensive system for processing and summarizing legal documents using advanced language models. This project includes tools for dataset preparation, model fine-tuning, document processing, and a web application for interactive use.

## Features

- **Document Processing**: Support for PDF and DOCX files
- **AI-Powered Summarization**: Utilizes a custom Ollama model (`legal-summarizer`) for accurate legal document summarization
- **Summary Verification**: Uses the `gemma3:4b` Ollama model to verify and assess summary accuracy
- **Batch Processing**: Process multiple documents in parallel
- **Export Capabilities**: Save summaries in DOCX format
- **Experiment Tracking**: Integration with Weights & Biases for training monitoring
- **Web Application**: Upload, summarize, and verify documents via a user-friendly web interface

## Project Structure

```
legal_ai/
├── datasets/                    # Directory for processed datasets
├── billsum_llama3_finetuned/    # Directory for fine-tuned models
├── documents/                   # Input documents directory
├── output/                      # Generated summaries directory
├── refined/                     # Summaries in DOCX format
├── app.py                       # Flask web application
├── main.py                      # Main application script
├── gemma_analyzer.py            # Summary analysis/verification logic
├── requirements.txt             # Project dependencies
├── templates/
│   └── index.html               # Web app frontend
├── ...                          # Other scripts, configs, or files you may add
```

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended 16GB+ VRAM)
- [Ollama](https://ollama.ai/) installed and running
- Hugging Face account (for accessing models)
- Weights & Biases account (for experiment tracking)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd legal-ai
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Web Application Usage

### 1. Start Ollama and Load Models

Make sure you have [Ollama](https://ollama.ai/) installed and running, and that you have created or pulled the following models:
```bash
ollama run legal-summarizer
ollama pull gemma3:4b
```

### 2. Start the Flask Web App

```bash
python app.py
```

Visit [http://localhost:5000](http://localhost:5000) in your browser.

### 3. Upload and Analyze Documents

- Upload PDF or DOCX files via the web interface.
- The app will generate a summary using your custom `legal-summarizer` model.
- You can then check the summary's accuracy using the `gemma3:4b` model.
- The results will show:
  - **Assessment** (GREAT, GOOD, FAIR, or NOT SO ACCURATE)
  - **Similarity Score**
  - **Explanation**

## Usage (Fine-tuning and CLI)

### 1. Dataset Preparation

Prepare the BillSum dataset for fine-tuning:
```bash
python prepare_legal_dataset.py
```

This will:
- Download the BillSum dataset
- Process and clean the data
- Save the processed dataset to `legal_ai/datasets/billsum_data.json`

### 2. Model Fine-tuning

Fine-tune the Llama 3 model:
```bash
python fine_tune_llama3.py
```

This will:
- Load the prepared dataset
- Fine-tune the model using LoRA
- Save the fine-tuned model to `legal_ai/billsum_llama3_finetuned`

### 3. Document Processing (CLI)

Process legal documents:
```bash
python main.py [path_to_documents]
```

If no path is provided, the script will use the default `documents` directory.

## Configuration

The system uses environment variables for configuration:

- `DEFAULT_DIRECTORY`: Directory for input documents (default: "documents")
- `OUTPUT_DIRECTORY`: Directory for generated summaries (default: "output")

## Output Format

The system generates summaries in the following format:

```
Title: [Title of the case or document]
Number: [Document or case number]
Date: [Date of the document or ruling]
Parties: [Names of individuals or entities involved]
Issues: [Primary legal questions or matters addressed]
Ruling: [Outcome or court decision]
Declarations: [Official statements or legal findings]
Precedents: [Legal precedents cited]
Summary: [Brief summary of the document]
```

## Requirements

Install dependencies with:
```bash
pip install -r requirements.txt
```

Your `requirements.txt` should include:
```
Flask>=2.0.0
Werkzeug>=2.0.0
PyPDF2>=3.0.0
langchain-community>=0.0.21
sentence-transformers>=2.2.2
requests>=2.31.0
numpy>=1.24.0
python-docx>=0.8.11
pdf2image>=1.16.3
pytesseract>=0.3.10
# ... (other dependencies as needed)
```

## Performance Considerations

- GPU Requirements: Minimum 16GB VRAM recommended for fine-tuning
- Memory Usage: Adjust batch size and gradient accumulation steps based on available VRAM
- Processing Speed: Multi-threaded document processing for improved performance

## Troubleshooting

1. **CUDA Errors**:
   - Ensure CUDA is properly installed
   - Check GPU compatibility
   - Verify VRAM availability

2. **Model Loading Issues**:
   - Verify Hugging Face token
   - Check internet connection
   - Ensure sufficient disk space

3. **Document Processing Errors**:
   - Verify file formats (PDF/DOCX)
   - Check file permissions
   - Ensure sufficient system memory

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Specify your license here]

## Acknowledgments

- BillSum dataset
- Hugging Face Transformers
- Unsloth for model optimization
- Weights & Biases for experiment tracking 