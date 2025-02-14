from flask import Flask, request, jsonify, render_template
import fitz  # PyMuPDF
from PIL import Image
import os
import google.generativeai as genai
import nltk
import torch
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer




app = Flask(__name__)

# Configure Gemini API
genai.configure(api_key="AIzaSyAawh0tRqyCOsyz7x9GxVbV_tkUzBsZ59s")

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
negation_words = {"not", "never", "no", "none", "cannot", "n't"}  # Add more if needed

# SBERT Model for Similarity
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# Cross-Encoder for Contextual Understanding
cross_encoder_model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/stsb-roberta-large")
cross_encoder_tokenizer = AutoTokenizer.from_pretrained("cross-encoder/stsb-roberta-large")

# Directory to save uploaded files
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Storage for teacher answers
teacher_answers = {}  # {page_number: text}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Preprocessing Function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Convert to lowercase & tokenize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Remove stopwords & lemmatize
    return " ".join(tokens)

# Function to Check for Negation
def contains_negation(text):
    tokens = set(word_tokenize(text.lower()))
    return any(word in negation_words for word in tokens)

def bert_similarity(student_answer, original_answer):
    # Preprocess text
    student_answer_clean = preprocess_text(student_answer)
    original_answer_clean = preprocess_text(original_answer)

    # SBERT similarity
    emb1 = sbert_model.encode(student_answer_clean, convert_to_tensor=True)
    emb2 = sbert_model.encode(original_answer_clean, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(emb1, emb2).item() * 100  # Convert to percentage

    # Contextual Understanding with Cross-Encoder
    inputs = cross_encoder_tokenizer(student_answer_clean, original_answer_clean, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = cross_encoder_model(**inputs).logits
    context_score = torch.sigmoid(logits).item() * 100  # Convert to percentage

    # Negation Handling
    student_has_negation = contains_negation(student_answer)
    original_has_negation = contains_negation(original_answer)

    if student_has_negation != original_has_negation:  # If one has negation and the other doesn't
        similarity *= 0.5  # Reduce similarity by 50%
        context_score *= 0.5  # Reduce context score by 50%

    return similarity, context_score

def extract_text_from_image(image):

    model = genai.GenerativeModel("gemini-1.5-flash")

    # Send the image to Gemini Vision API
    prompt = "Extract and return the handwritten text from this image:"
    response = model.generate_content([prompt, image])

    # Return the extracted text
    return response.text.strip()

def extract_text_from_pdf(pdf_path):
    """
    Extracts handwritten text from a multi-page PDF and returns it as a list.
    Each page's text is stored at its corresponding index.
    """
    doc = fitz.open(pdf_path)
    extracted_text_list = []

    for i, page in enumerate(doc):
        print(f"Processing Page {i + 1}...")

        # Convert PDF page to an image
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

        # Extract text from the image
        text = extract_text_from_image(img)
        extracted_text_list.append(text.strip())  # Strip to remove unnecessary spaces

    return extracted_text_list

@app.route('/')
def index():
    """
    Render the homepage with options to upload Teacher and Student PDFs.
    """
    return render_template("index.html")  # Create an HTML form for uploading files

@app.route('/upload/teacher', methods=['POST'])
def upload_teacher_pdf():
    """
    Endpoint to upload teacher's PDF and store answers page-wise.
    """
    if 'pdf' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    pdf_file = request.files['pdf']
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_file.filename)
    pdf_file.save(pdf_path)

    # Extract text from the uploaded PDF
    global teacher_answers
    teacher_answers = extract_text_from_pdf(pdf_path)

    return jsonify({"message": "Teacher answers uploaded successfully", "pages": len(teacher_answers)})

@app.route('/upload/student', methods=['POST'])
def upload_student_pdf():
    """
    Endpoint to upload a student's PDF, extract answers, and compare with teacher's answers.
    """
    if 'pdf' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    pdf_file = request.files['pdf']
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_file.filename)
    pdf_file.save(pdf_path)

    # Extract text from the uploaded student PDF
    extracted_answers = extract_text_from_pdf(pdf_path)

    # Compare with teacher's answers
    comparisons = {}
    for page, (student_text, teacher_text) in enumerate(zip(extracted_answers, teacher_answers), start=1):
        similarity_score, contextual_score = bert_similarity(student_text, teacher_text)
        comparisons[page] = {
            "student_text": student_text,
            "teacher_text": teacher_text,
            "similarity_score": similarity_score,
            "contextual_score": contextual_score,
        }


    # Return the comparisons for display
    return render_template("result.html", comparisons=comparisons)


if __name__ == '__main__':
    app.run(debug=True)
