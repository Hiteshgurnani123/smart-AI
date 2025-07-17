import os
import json
import joblib
import re
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import pymupdf  # PyMuPDF
import docx2txt
import google.generativeai as genai
from fpdf import FPDF
import markdown2 # PDF me formatting ke liye

# --- CONFIGURATION ---
# Apni Gemini API Key yahan daalein
# IMPORTANT: Apni key yahan daalein
# Agar aapke paas API key nahi hai, to ise khaali chhod dein: ''
GEMINI_API_KEY = 'YOUR_GEMINI_API_KEY'

# --- INITIALIZATION ---
# Flask app initialize karein
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Gemini API ko configure karein
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# --- HELPER FUNCTIONS (Must be defined before loading models) ---

def custom_tokenizer(text):
    """
    This function is required for joblib to load the TF-IDF vectorizer.
    It must be identical to the tokenizer used during the model's training.
    It splits a comma-separated string of skills.
    """
    return text.split(', ')

# --- LOAD MODELS AND DATA (Ek hi baar load hoga jab app start hogi) ---
try:
    # Machine Learning models aur encoders load karein
    model = joblib.load('random_forest_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    edu_encoder = joblib.load('edu_encoder.pkl')
    loc_encoder = joblib.load('loc_encoder.pkl')
    label_encoder = joblib.load('label_encoder.pkl')

    # Job roles aur skills ka data load karein
    with open('job_roles_skills.json', 'r') as f:
        job_roles_skills = json.load(f)
except FileNotFoundError as e:
    print(f"ERROR: Zaroori file nahi mili: {e}. Sunishchit karein ki sabhi .pkl aur .json files maujood hain.")
    # App ko crash hone se rokne ke liye, aap yahan exit kar sakte hain ya default values set kar sakte hain
    model, tfidf, edu_encoder, loc_encoder, label_encoder, job_roles_skills = (None,)*6


def extract_text_from_resume(filepath):
    """PDF ya DOCX file se text nikalne ke liye function."""
    text = ""
    try:
        if filepath.endswith('.pdf'):
            doc = pymupdf.open(filepath)
            for page in doc:
                text += page.get_text()
        elif filepath.endswith('.docx'):
            text = docx2txt.process(filepath)
    except Exception as e:
        print(f"Error extracting text from {filepath}: {e}")
    return text

def extract_skills_from_text(text, all_skills):
    """Diye gaye text se skills nikalne ke liye function."""
    extracted_skills = set()
    text_lower = text.lower()
    for skill in all_skills:
        # Whole word matching ke liye regular expression
        if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text_lower):
            extracted_skills.add(skill)
    return list(extracted_skills)

def generate_skill_gap_chart(user_skills, required_skills, predicted_title):
    """Pie chart banane ke liye function jo skill gap dikhata hai."""
    import matplotlib
    matplotlib.use('Agg') # GUI backend ke bina chalane ke liye
    import matplotlib.pyplot as plt

    user_skill_set = set(user_skills)
    required_skill_set = set(required_skills)

    matched_skills_count = len(user_skill_set.intersection(required_skill_set))
    missing_skills_count = len(required_skill_set - user_skill_set)

    # Agar koi required skill nahi hai, to chart na banayein
    if not required_skill_set:
        return None

    labels = ['Aapke Paas Hain', 'Zaroori Hain (Missing)']
    sizes = [matched_skills_count, missing_skills_count]
    colors = ['#28a745', '#ffc107'] # Green and Yellow
    explode = (0, 0.1) if missing_skills_count > 0 else (0, 0)

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=140, colors=colors, textprops={'fontsize': 14, 'color': 'black'})
    ax1.axis('equal')
    plt.title(f'{predicted_title} ke liye Skill Gap', fontsize=16, pad=20)
    
    chart_path = os.path.join(app.config['UPLOAD_FOLDER'], "skill_gap_chart.png")
    plt.savefig(chart_path)
    plt.close()
    return os.path.basename(chart_path)

def get_roadmap_from_gemini(job_title):
    """Gemini API se career roadmap generate karne ke liye function."""
    if not GEMINI_API_KEY or not genai:
        return "Gemini API key configure nahi hai. Kripya `app.py` me apni key daalein."
    try:
        # *** BUG FIX 3: Sahi model ka naam istemal karein ***
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = f"Act as a career coach. Create a detailed, step-by-step career roadmap for someone who wants to become a '{job_title}'. The response should be in Hinglish (Hindi written in English script). The output must be in Markdown format. Include sections for: Foundational Skills (Week 1-4), Intermediate Skills (Week 5-8), Advanced Topics (Week 9-12), and Project Building (Week 13-16). For each skill, suggest 1-2 popular online resources (like YouTube channels, Udemy courses, or official docs)."
        response = model.generate_content(prompt)
        # Markdown ko HTML me convert karein
        return markdown2.markdown(response.text)
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return f"Roadmap generate karte samay error aa gaya: {e}. Kripya check karein ki aapka API key sahi hai aur model ka naam (`gemini-1.5-flash-latest`) astitva mein hai."


# --- FLASK ROUTES ---

@app.route('/')
def index():
    """Home page render karta hai."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Resume upload, process, aur result page render karta hai."""
    if not all([model, tfidf, edu_encoder, loc_encoder, label_encoder, job_roles_skills]):
        return "Server theek se configure nahi hua hai. Kripya admin se sampark karein.", 500

    if 'resume' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['resume']
    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # 1. Resume se text aur skills extract karein
        resume_text = extract_text_from_resume(filepath)
        all_possible_skills = {skill for skills in job_roles_skills.values() for skill in skills}
        user_skills = extract_skills_from_text(resume_text, all_possible_skills)

        # 2. Model ke liye input taiyyar karein
        skills_tfidf = tfidf.transform([', '.join(user_skills)]).toarray()
        
        # Default values (aap ise form se bhi le sakte hain)
        education_level = 'B.Tech'
        location = 'Bangalore'
        experience = 2 # years

        education_encoded = edu_encoder.transform([[education_level]])
        location_encoded = loc_encoder.transform([[location]])
        experience_reshaped = np.array([[experience]])
        
        X_new = np.hstack((skills_tfidf, education_encoded, location_encoded, experience_reshaped))

        # 3. Job title predict karein
        prediction_index = model.predict(X_new)[0]
        predicted_job_title = label_encoder.inverse_transform([prediction_index])[0]

        # 4. Required aur Missing skills calculate karein
        # *** BUG FIX 1: Missing skills ko sahi se calculate karein ***
        required_skills = job_roles_skills.get(predicted_job_title, [])
        missing_skills = list(set(required_skills) - set(user_skills))
        
        # 5. Skill gap chart generate karein
        # *** TYPO FIX: Corrected variable name from predicted_jo_title to predicted_job_title ***
        chart_filename = generate_skill_gap_chart(user_skills, required_skills, predicted_job_title)
        chart_url = url_for('static_file', filename=chart_filename) if chart_filename else None

        # 6. Result page ko saare data ke saath render karein
        return render_template(
            'result.html',
            job_title=predicted_job_title,
            user_skills=user_skills,
            missing_skills=missing_skills,
            chart_url=chart_url
        )

@app.route('/roadmap')
def roadmap():
    """Gemini se roadmap generate karke page par dikhata hai."""
    job_title = request.args.get('job_title', 'Software Developer')
    roadmap_html_content = get_roadmap_from_gemini(job_title)
    return render_template('roadmap.html', job_title=job_title, roadmap=roadmap_html_content)

@app.route('/download_roadmap')
def download_roadmap():
    """Roadmap ko PDF format me download karne ke liye."""
    job_title = request.args.get('job_title', 'Software Developer')
    
    # *** BUG FIX 4: PDF banane ke liye roadmap content dobara generate karein ***
    roadmap_html = get_roadmap_from_gemini(job_title)
    # HTML se saaf text nikalein (PDF ke liye)
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(roadmap_html, 'html.parser')
    roadmap_text = soup.get_text()

    pdf = FPDF()
    pdf.add_page()
    pdf.add_font('Poppins', '', 'Poppins-Regular.ttf', uni=True)
    pdf.set_font('Poppins', '', 16)
    pdf.cell(0, 10, txt=f"{job_title} Roadmap", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font('Poppins', '', 11)
    pdf.multi_cell(0, 8, txt=roadmap_text)
    
    pdf_filename = f'{job_title.replace(" ", "_")}_roadmap.pdf'
    pdf_output_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
    pdf.output(pdf_output_path)

    return send_from_directory(app.config['UPLOAD_FOLDER'], pdf_filename, as_attachment=True)
    
@app.route('/uploads/<filename>')
def static_file(filename):
    """Static folder se files serve karne ke liye (jaise chart image)."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    # Zaroori libraries install karne ke liye check
    try:
        import bs4
        import markdown2
    except ImportError:
        print("\n--- Zaroori Package Install Karein ---")
        print("pip install beautifulsoup4 markdown2\n")

    app.run(debug=True)
