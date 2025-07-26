import requests
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class SummaryAnalyzer:
    def __init__(self, model_name="gemma3:4b"):
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model_name = model_name

        if not self.model_name:
            logger.warning("No model name specified. Please provide a model available in Ollama.")
    
    def calculate_similarity(self, text1, text2):
        embedding1 = self.sentence_model.encode(text1)
        embedding2 = self.sentence_model.encode(text2)
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        return float(similarity)

    def generate_analysis(self, prompt):
        try:
            logger.info(f"Generating response using model: {self.model_name}")
            logger.debug(f"Prompt: {prompt[:100]}...")  # Preview first 100 characters
            
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_ctx": 4096
                },
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            logger.error(f"Error generating analysis: {str(e)}")
            return "Error generating analysis. Please try again."

    def generate_analysis_paragraph(self, original_text, summary):
        prompt = f"""
Compare the following original document and summary. Write a 2-3 sentence factual analysis of how accurately the summary reflects the original document. Do not provide any other information.

Original Document:
{original_text}

Generated Summary:
{summary}

Analysis:
"""
        return self.generate_analysis(prompt)

    def generate_assessment(self, similarity_score, analysis_paragraph):
        prompt = f"""
Given the following similarity score and analysis, choose ONE label for the summary's accuracy: GREAT, GOOD, FAIR, or NOT SO ACCURATE. Only output the label.

Similarity Score: {similarity_score:.2f}
Analysis: {analysis_paragraph}

Assessment:
"""
        return self.generate_analysis(prompt).strip().upper()

    def generate_explanation(self, similarity_score, analysis_paragraph, assessment):
        prompt = f"""
Given the following similarity score, analysis, and assessment, write a 2-3 line explanation justifying the assessment. Only output the explanation.

Similarity Score: {similarity_score:.2f}
Analysis: {analysis_paragraph}
Assessment: {assessment}

Explanation:
"""
        return self.generate_analysis(prompt).strip()

    def analyze_summary(self, original_text, summary):
        similarity_score = self.calculate_similarity(original_text, summary)
        analysis_paragraph = self.generate_analysis_paragraph(original_text, summary)
        assessment = self.generate_assessment(similarity_score, analysis_paragraph)
        explanation = self.generate_explanation(similarity_score, analysis_paragraph, assessment)

        valid_assessments = ['GREAT', 'GOOD', 'FAIR', 'NOT SO ACCURATE']
        if assessment not in valid_assessments:
            logger.warning(f"Invalid assessment received: {assessment}")
            assessment = 'Not Assessed'

        return {
            'similarity_score': similarity_score,
            'analysis': analysis_paragraph,
            'assessment': assessment,
            'explanation': explanation
        }

    def test_connection(self):
        """Utility function to check if Ollama is reachable and the model is ready."""
        try:
            logger.info("Testing connection to Ollama...")
            test_prompt = "Say hello!"
            test_response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model_name,
                    "prompt": test_prompt,
                    "stream": False
                },
                headers={"Content-Type": "application/json"}
            )
            test_response.raise_for_status()
            output = test_response.json().get("response", "")
            logger.info("Connection successful.")
            logger.debug(f"Sample Output: {output}")
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
