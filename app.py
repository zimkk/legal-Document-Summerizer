from flask import Flask, render_template, request, jsonify
import os
import main
from werkzeug.utils import secure_filename
import threading
import queue
import time
import glob
from gemma_analyzer import SummaryAnalyzer
import json
import logging
from docx import Document

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the summary analyzer
analyzer = SummaryAnalyzer()

# Queue for processing status
processing_queue = queue.Queue()
current_status = {"message": "Processing started", "progress": 0}

logger = logging.getLogger(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf', 'docx'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files[]')
    if not files:
        return jsonify({'error': 'No files selected'}), 400

    uploaded_files = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            uploaded_files.append(file_path)

    if not uploaded_files:
        return jsonify({'error': 'No valid files uploaded'}), 400

    try:
        for file_path in uploaded_files:
            thread = threading.Thread(target=process_file_async, args=(file_path,))
            thread.daemon = True
            thread.start()
        return jsonify({'message': 'Processing document...'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def process_file_async(file_path):
    try:
        logger.info(f"Starting async processing of file: {file_path}")
        processing_queue.put({"message": f"Processing file: {os.path.basename(file_path)}", "progress": 0})
        
        # Step 1: Process the file and get the summary
        logger.info("Step 1: Generating summary...")
        summary = main.process_file(file_path)
        
        # Get the summary file path
        base_name = os.path.basename(file_path)
        name_without_ext = os.path.splitext(base_name)[0]
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        summary_path = os.path.join('refined', f"{name_without_ext}_Summary_{timestamp}.docx")
        
        processing_queue.put({
            "message": "Summary generated successfully",
            "progress": 100,
            "summary_path": summary_path,
            "original_file": file_path  # Pass the original file path
        })
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        processing_queue.put({"message": f"Error: {str(e)}", "progress": -1})

@app.route('/analyze', methods=['POST'])
def analyze_summary():
    try:
        data = request.json
        if not data or 'original_file' not in data or 'summary_path' not in data:
            return jsonify({'error': 'Missing file paths'}), 400

        original_file = data['original_file']
        summary_path = data['summary_path']

        if not os.path.exists(original_file):
            return jsonify({'error': 'Original file not found'}), 404
        if not os.path.exists(summary_path):
            return jsonify({'error': 'Summary file not found'}), 404

        # Read the original text
        logger.info(f"Reading original file: {original_file}")
        if original_file.endswith('.pdf'):
            with open(original_file, 'rb') as f:
                original_text = main.extract_text_from_pdf(original_file)
        elif original_file.endswith('.docx'):
            with open(original_file, 'rb') as f:
                original_text = main.extract_text_from_docx(original_file)
        else:
            return jsonify({'error': 'Unsupported file type'}), 400

        # Read the summary
        logger.info(f"Reading summary file: {summary_path}")
        doc = Document(summary_path)
        summary_text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])

        # Analyze the summary
        logger.info("Analyzing summary...")
        analysis = analyzer.analyze_summary(original_text, summary_text)
        
        # Save the analysis results
        output_path = os.path.join('output', f"{os.path.basename(original_file)}_analysis.json")
        os.makedirs('output', exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=4)

        return jsonify({
            'message': 'Analysis completed successfully',
            'analysis': analysis
        })
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/status')
def get_status():
    try:
        status = processing_queue.get_nowait()
        return jsonify(status)
    except queue.Empty:
        return jsonify(current_status)

@app.route('/analysis/<filename>')
def get_analysis(filename):
    try:
        analysis_path = os.path.join('output', f"{filename}_analysis.json")
        if os.path.exists(analysis_path):
            with open(analysis_path, 'r', encoding='utf-8') as f:
                return jsonify(json.load(f))
        return jsonify({'error': 'Analysis not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 