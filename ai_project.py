#!/usr/bin/env python3
"""
Educational Generative AI App using IBM Granite 3.2 2B Instruct Model
Features: Concept Explanation, Quiz Generator, Ask From Files, Grammar Correction, Prompt Enhancement
Built for Google Colab with Gradio UI
"""

import os
import gc
import re
import io
import base64
import warnings
from typing import Optional, Tuple, List
from difflib import SequenceMatcher, get_close_matches
import gradio as gr

# Suppress warnings
warnings.filterwarnings("ignore")

# Install required packages
def install_requirements():
    """Install required packages for the application"""
    import subprocess
    import sys
    
    packages = [
        "transformers>=4.35.0",
        "torch>=2.0.0",
        "gradio>=4.0.0",
        "accelerate>=0.24.0",
        "PyPDF2>=3.0.0",
        "pytesseract>=0.3.10",
        "pillow>=10.0.0",
        "requests>=2.31.0",
        "huggingface_hub>=0.19.0"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to install {package}: {e}")
    
    # Install tesseract for OCR
    try:
        subprocess.check_call(["apt-get", "update"])
        subprocess.check_call(["apt-get", "install", "-y", "tesseract-ocr"])
    except subprocess.CalledProcessError:
        print("Warning: Could not install tesseract-ocr via apt-get")

# Install requirements
print("Installing required packages...")
install_requirements()

# Import libraries after installation
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import PyPDF2
import pytesseract
from PIL import Image
import requests
import json
from huggingface_hub import InferenceClient

class EducationalAIApp:
    def __init__(self):
        self.model_name = "ibm-granite/granite-3.2-2b-instruct"
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.inference_client = None
        self.use_local_model = False
        self.hf_token = None
        
        # Educational topics database for spell checking
        self.educational_topics = {
            # Science
            'photosynthesis', 'mitosis', 'meiosis', 'dna', 'rna', 'evolution', 'genetics', 'ecosystem',
            'gravity', 'magnetism', 'electricity', 'thermodynamics', 'quantum mechanics', 'relativity',
            'chemistry', 'biology', 'physics', 'astronomy', 'geology', 'meteorology',
            'photosynthesis', 'respiration', 'digestion', 'circulation', 'nervous system',
            'atomic structure', 'periodic table', 'chemical reactions', 'acids', 'bases',
            
            # Mathematics
            'algebra', 'geometry', 'trigonometry', 'calculus', 'statistics', 'probability',
            'quadratic equations', 'linear equations', 'logarithms', 'exponents', 'fractions',
            'derivatives', 'integrals', 'matrices', 'vectors', 'functions', 'polynomials',
            
            # History
            'world war', 'civil war', 'revolution', 'renaissance', 'enlightenment',
            'ancient rome', 'ancient greece', 'egypt', 'mesopotamia', 'feudalism',
            'colonialism', 'imperialism', 'democracy', 'monarchy', 'republic',
            
            # Literature & Language
            'shakespeare', 'poetry', 'novel', 'metaphor', 'simile', 'alliteration',
            'grammar', 'syntax', 'phonetics', 'linguistics', 'etymology',
            'prose', 'verse', 'rhetoric', 'narrative', 'character development',
            
            # Geography
            'continents', 'countries', 'capitals', 'mountains', 'rivers', 'oceans',
            'climate', 'weather', 'topography', 'cartography', 'longitude', 'latitude',
            
            # Economics & Social Studies
            'economics', 'supply and demand', 'inflation', 'recession', 'gdp',
            'government', 'constitution', 'democracy', 'citizenship', 'civil rights',
            
            # Computer Science
            'programming', 'algorithms', 'data structures', 'machine learning',
            'artificial intelligence', 'databases', 'networks', 'software engineering'
        }
        
        # Initialize the model
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize the model - try local first, then fallback to HF API"""
        print("Initializing IBM Granite 3.2 2B Instruct model...")
        
        # Check for HF token
        self.hf_token = os.getenv("HF_TOKEN")
        if not self.hf_token:
            print("No HF_TOKEN found in environment. You may need to provide it for API fallback.")
        
        # Try to load model locally
        try:
            print("Attempting to load model locally...")
            
            # Check GPU availability
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.hf_token,
                trust_remote_code=True
            )
            
            # Load model with appropriate settings
            model_kwargs = {
                "token": self.hf_token,
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
                "low_cpu_mem_usage": True
            }
            
            if device == "cuda":
                model_kwargs["device_map"] = "auto"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                trust_remote_code=True
            )
            
            self.use_local_model = True
            print("✅ Model loaded locally successfully!")
            
        except Exception as e:
            print(f"❌ Failed to load model locally: {e}")
            print("🔄 Falling back to Hugging Face Inference API...")
            
            if not self.hf_token:
                self.hf_token = input("Please enter your Hugging Face token for API access: ").strip()
            
            if self.hf_token:
                try:
                    self.inference_client = InferenceClient(
                        model=self.model_name,
                        token=self.hf_token
                    )
                    print("✅ Inference API client initialized successfully!")
                except Exception as api_error:
                    print(f"❌ Failed to initialize Inference API: {api_error}")
                    raise Exception("Could not initialize model locally or via API")
            else:
                raise Exception("No HF token provided for API fallback")
    
    def generate_response(self, prompt: str, max_length: int = 512) -> str:
        """Generate response using local model or API"""
        try:
            if self.use_local_model and self.pipeline:
                # Local model generation
                messages = [{"role": "user", "content": prompt}]
                
                response = self.pipeline(
                    messages,
                    max_new_tokens=max_length,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                return response[0]["generated_text"][-1]["content"]
            
            elif self.inference_client:
                # API generation
                response = self.inference_client.text_generation(
                    prompt,
                    max_new_tokens=max_length,
                    temperature=0.7,
                    do_sample=True,
                    return_full_text=False
                )
                return response
            
            else:
                return "❌ Model not available. Please check your setup."
                
        except Exception as e:
            return f"❌ Error generating response: {str(e)}"
    
    def is_educational_prompt(self, prompt: str) -> bool:
        """Check if the prompt is educational in nature"""
        educational_keywords = [
            'explain', 'learn', 'understand', 'concept', 'definition', 'how does',
            'what is', 'why does', 'theory', 'principle', 'study', 'academic',
            'education', 'teach', 'lesson', 'course', 'subject', 'topic',
            'knowledge', 'information', 'science', 'math', 'history', 'literature',
            'biology', 'chemistry', 'physics', 'geography', 'philosophy'
        ]
        
        prompt_lower = prompt.lower()
        return any(keyword in prompt_lower for keyword in educational_keywords)
    
    def check_spelling_and_suggest(self, user_input: str) -> Tuple[str, List[str]]:
        """Check spelling and suggest corrections for educational topics"""
        words = re.findall(r'\b\w+\b', user_input.lower())
        suggestions = []
        corrected_input = user_input.lower()
        
        for word in words:
            if len(word) > 3:  # Only check words longer than 3 characters
                # Find close matches in educational topics
                close_matches = get_close_matches(
                    word, 
                    self.educational_topics, 
                    n=3, 
                    cutoff=0.6
                )
                
                if close_matches and word not in self.educational_topics:
                    # Check if it's a reasonable match (similarity > 0.7)
                    best_match = close_matches[0]
                    similarity = SequenceMatcher(None, word, best_match).ratio()
                    
                    if similarity > 0.7:
                        suggestions.append(f"'{word}' → '{best_match}'")
                        corrected_input = corrected_input.replace(word, best_match)
        
        return corrected_input, suggestions
    
    def format_suggestion_message(self, original_input: str, suggestions: List[str]) -> str:
        """Format spelling suggestion message"""
        if not suggestions:
            return ""
        
        suggestion_text = "🔍 **Spelling Suggestions:**\n"
        suggestion_text += "Did you mean:\n"
        for suggestion in suggestions[:3]:  # Limit to top 3 suggestions
            suggestion_text += f"• {suggestion}\n"
        suggestion_text += "\n---\n\n"
        return suggestion_text
    
    def explain_concept(self, concept: str) -> str:
        """Explain educational concepts with educational content filter"""
        if not concept.strip():
            return "Please enter a concept to explain."
        
        # Check spelling and get suggestions
        corrected_concept, suggestions = self.check_spelling_and_suggest(concept)
        suggestion_message = self.format_suggestion_message(concept, suggestions)
        
        # Use corrected concept for educational check
        if not self.is_educational_prompt(corrected_concept):
            return """
            🚫 **Educational Content Only**
            
            This feature is designed specifically for educational purposes. Please ask about:
            - Academic concepts and theories
            - Scientific principles
            - Mathematical concepts
            - Historical events
            - Literature and language
            - Any learning-related topic
            
            Please rephrase your request to focus on educational content.
            """
        
        # Use corrected concept for generation
        concept_to_use = corrected_concept if suggestions else concept
        
        prompt = f"""You are an educational AI assistant. Provide a clear, comprehensive explanation of the following concept in an educational context:

Concept: {concept_to_use}

Please explain this concept in a way that is:
1. Easy to understand
2. Educationally valuable
3. Includes key points and examples
4. Suitable for learning purposes

Explanation:"""
        
        response = self.generate_response(prompt, max_length=400)
        
        # Add spelling suggestions to the beginning if any
        if suggestions:
            return suggestion_message + response
        
        return response
    
    def generate_quiz(self, topic: str, num_questions: int = 5) -> str:
        """Generate educational quizzes with educational content filter"""
        if not topic.strip():
            return "Please enter a topic for the quiz."
        
        # Check spelling and get suggestions
        corrected_topic, suggestions = self.check_spelling_and_suggest(topic)
        suggestion_message = self.format_suggestion_message(topic, suggestions)
        
        # Use corrected topic for educational check
        if not self.is_educational_prompt(corrected_topic):
            return """
            🚫 **Educational Content Only**
            
            The quiz generator is designed for educational topics only. Please request quizzes about:
            - Academic subjects
            - Scientific concepts
            - Mathematical topics
            - Historical periods
            - Literature and language
            - Any educational subject
            
            Please specify an educational topic for your quiz.
            """
        
        # Use corrected topic for generation
        topic_to_use = corrected_topic if suggestions else topic
        
        prompt = f"""Create an educational quiz about: {topic_to_use}

Generate exactly {num_questions} multiple-choice questions with the following format:
- Each question should test understanding of key concepts
- Provide 4 options (A, B, C, D) for each question
- Include the correct answer
- Make questions educational and appropriate for learning

Topic: {topic_to_use}
Number of questions: {num_questions}

Quiz:"""
        
        response = self.generate_response(prompt, max_length=600)
        
        # Add spelling suggestions to the beginning if any
        if suggestions:
            return suggestion_message + response
        
        return response
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            return f"Error extracting PDF text: {str(e)}"
    
    def extract_text_from_image(self, image_file) -> str:
        """Extract text from image using OCR"""
        try:
            image = Image.open(image_file)
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            return f"Error extracting image text: {str(e)}"
    
    def ask_from_files(self, files, question: str) -> str:
        """Answer questions based on uploaded files"""
        if not files:
            return "Please upload at least one file (PDF, image, or text file)."
        
        if not question.strip():
            return "Please enter a question about the uploaded files."
        
        extracted_texts = []
        
        for file in files:
            try:
                if file.name.lower().endswith('.pdf'):
                    text = self.extract_text_from_pdf(file)
                elif file.name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    text = self.extract_text_from_image(file)
                elif file.name.lower().endswith('.txt'):
                    text = file.read().decode('utf-8')
                else:
                    continue
                
                if text and not text.startswith("Error"):
                    extracted_texts.append(f"File: {file.name}\nContent:\n{text}\n")
            except Exception as e:
                extracted_texts.append(f"File: {file.name}\nError: {str(e)}\n")
        
        if not extracted_texts:
            return "Could not extract text from any of the uploaded files."
        
        combined_text = "\n".join(extracted_texts)
        
        prompt = f"""Based on the following extracted text from uploaded files, please answer the user's question:

Extracted Content:
{combined_text[:3000]}  # Limit context length

User Question: {question}

Please provide a comprehensive answer based on the content from the files:"""
        
        return self.generate_response(prompt, max_length=400)
    
    def correct_grammar(self, text: str) -> str:
        """Correct grammar and sentence structure"""
        if not text.strip():
            return "Please enter text to correct."
        
        prompt = f"""Please correct the grammar, spelling, and sentence structure of the following text. Maintain the original meaning while improving clarity and correctness:

Original text: {text}

Corrected text:"""
        
        return self.generate_response(prompt, max_length=300)
    
    def enhance_prompt(self, user_prompt: str) -> str:
        """Enhance and improve user prompts"""
        if not user_prompt.strip():
            return "Please enter a prompt to enhance."
        
        prompt = f"""Improve and enhance the following prompt to make it more effective, clear, and specific. The enhanced prompt should be better structured and more likely to generate high-quality responses:

Original prompt: {user_prompt}

Enhanced prompt:"""
        
        return self.generate_response(prompt, max_length=200)
    
    def create_gradio_interface(self):
        """Create the Gradio interface"""
        with gr.Blocks(title="Educational AI Assistant", theme=gr.themes.Soft()) as app:
            gr.HTML("""
            <div style="text-align: center; margin-bottom: 20px;">
                <h1>🎓 Educational AI Assistant</h1>
                <p>Powered by IBM Granite 3.2 2B Instruct Model</p>
                <p><em>Focused on educational content and learning support</em></p>
            </div>
            """)
            
            with gr.Tabs():
                # Concept Explanation Tab
                with gr.TabItem("📚 Concept Explanation"):
                    gr.Markdown("### Explain Educational Concepts")
                    gr.Markdown("*This feature only accepts educational topics and concepts. It will auto-correct spelling mistakes!*")
                    
                    concept_input = gr.Textbox(
                        label="Enter a concept to explain",
                        placeholder="e.g., fotosynthesis, quadretic equations, World War II... (spelling will be auto-corrected)",
                        lines=2
                    )
                    concept_button = gr.Button("Explain Concept", variant="primary")
                    concept_output = gr.Textbox(
                        label="Explanation",
                        lines=10,
                        show_copy_button=True
                    )
                    
                    concept_button.click(
                        self.explain_concept,
                        inputs=[concept_input],
                        outputs=[concept_output]
                    )
                
                # Quiz Generator Tab
                with gr.TabItem("🧩 Quiz Generator"):
                    gr.Markdown("### Generate Educational Quizzes")
                    gr.Markdown("*This feature creates quizzes for educational topics only. Spelling mistakes will be auto-corrected!*")
                    
                    with gr.Row():
                        quiz_topic = gr.Textbox(
                            label="Quiz Topic",
                            placeholder="e.g., celular biology, algebre, American histery... (spelling will be auto-corrected)",
                            lines=2,
                            scale=3
                        )
                        quiz_questions = gr.Slider(
                            label="Number of Questions",
                            minimum=3,
                            maximum=10,
                            value=5,
                            step=1,
                            scale=1
                        )
                    
                    quiz_button = gr.Button("Generate Quiz", variant="primary")
                    quiz_output = gr.Textbox(
                        label="Generated Quiz",
                        lines=15,
                        show_copy_button=True
                    )
                    
                    quiz_button.click(
                        self.generate_quiz,
                        inputs=[quiz_topic, quiz_questions],
                        outputs=[quiz_output]
                    )
                
                # Ask From Files Tab
                with gr.TabItem("📄 Ask From Files"):
                    gr.Markdown("### Upload Files and Ask Questions")
                    gr.Markdown("*Upload PDF, image, or text files and ask questions about their content.*")
                    
                    file_upload = gr.File(
                        label="Upload Files (PDF, Images, Text)",
                        file_count="multiple",
                        file_types=[".pdf", ".txt", ".png", ".jpg", ".jpeg", ".bmp", ".tiff"]
                    )
                    file_question = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask a question about the uploaded files...",
                        lines=3
                    )
                    file_button = gr.Button("Ask Question", variant="primary")
                    file_output = gr.Textbox(
                        label="Answer",
                        lines=10,
                        show_copy_button=True
                    )
                    
                    file_button.click(
                        self.ask_from_files,
                        inputs=[file_upload, file_question],
                        outputs=[file_output]
                    )
                
                # Grammar Correction Tab
                with gr.TabItem("✏️ Grammar Correction"):
                    gr.Markdown("### Correct Grammar and Sentences")
                    
                    grammar_input = gr.Textbox(
                        label="Enter text to correct",
                        placeholder="Enter text with grammar or spelling errors...",
                        lines=5
                    )
                    grammar_button = gr.Button("Correct Text", variant="primary")
                    grammar_output = gr.Textbox(
                        label="Corrected Text",
                        lines=5,
                        show_copy_button=True
                    )
                    
                    grammar_button.click(
                        self.correct_grammar,
                        inputs=[grammar_input],
                        outputs=[grammar_output]
                    )
                
                # Prompt Enhancement Tab
                with gr.TabItem("🚀 Prompt Enhancer"):
                    gr.Markdown("### Enhance Your Prompts")
                    gr.Markdown("*Improve your prompts to get better AI responses.*")
                    
                    prompt_input = gr.Textbox(
                        label="Enter your prompt",
                        placeholder="Enter a prompt you want to improve...",
                        lines=4
                    )
                    prompt_button = gr.Button("Enhance Prompt", variant="primary")
                    prompt_output = gr.Textbox(
                        label="Enhanced Prompt",
                        lines=6,
                        show_copy_button=True
                    )
                    
                    prompt_button.click(
                        self.enhance_prompt,
                        inputs=[prompt_input],
                        outputs=[prompt_output]
                    )
            
            # Footer
            gr.HTML("""
            <div style="text-align: center; margin-top: 20px; padding: 10px; border-top: 1px solid #eee;">
                <p><strong>Model:</strong> IBM Granite 3.2 2B Instruct | <strong>Mode:</strong> {}</p>
                <p><em>Educational AI Assistant - Designed for learning and academic support</em></p>
            </div>
            """.format("Local GPU" if self.use_local_model else "Hugging Face API"))
        
        return app

# Main execution
def main():
    """Main function to run the application"""
    print("🎓 Educational AI Assistant")
    print("=" * 50)
    
    try:
        # Initialize the app
        app_instance = EducationalAIApp()
        
        # Create and launch Gradio interface
        demo = app_instance.create_gradio_interface()
        
        print("\n🚀 Launching Educational AI Assistant...")
        print("📱 The app will open in your browser automatically")
        
        # Launch with appropriate settings for Colab
        demo.launch(
            share=True,  # Create public link for Colab
            server_name="0.0.0.0",  # Allow external connections
            server_port=7860,  # Default Gradio port
            show_error=True,  # Show errors in interface
            quiet=False  # Show launch information
        )
        
    except Exception as e:
        print(f"❌ Error starting the application: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Make sure you have a stable internet connection")
        print("2. Check if you have enough GPU memory (for local model)")
        print("3. Verify your Hugging Face token (for API fallback)")
        print("4. Try restarting the runtime if issues persist")

if __name__ == "__main__":
    main()