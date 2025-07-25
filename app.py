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
import markdown2
from bs4 import BeautifulSoup


GEMINI_API_KEY = 'Your API key here' 



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Gemini API configure 
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"error at the time API configure: {e}")
        genai = None
else:
    print("API key not found.")
    genai = None

# --- HELPER FUNCTIONS 

def custom_tokenizer(text):
    """
    This function is required for joblib to load the TF-IDF vectorizer.
    It must be identical to the tokenizer used during the model's training.
    It splits a comma-separated string of skills.
    """
    return text.split(', ')

# --- LOAD MODELS AND DATA 
try:
    # Machine Learning models and encoders 
    model = joblib.load('random_forest_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    edu_encoder = joblib.load('edu_encoder.pkl')
    loc_encoder = joblib.load('loc_encoder.pkl')
    label_encoder = joblib.load('label_encoder.pkl')


    # Job roles and skills data load 
    with open('job_roles_skills.json', 'r') as f:
        job_roles_skills = json.load(f)
except FileNotFoundError as e:
    print(f"ERROR: important file not found : {e}. Make seur that all .pkl and .json files are uploaded.")

    model, tfidf, edu_encoder, loc_encoder, label_encoder, job_roles_skills = (None,)*6  # for stpping app crashes


def extract_text_from_resume(filepath):     # fuction for extracting text from resume 

    text = ""
    try:
        if filepath.lower().endswith('.pdf'):
            doc = pymupdf.open(filepath)
            for page in doc:
                text += page.get_text()
        elif filepath.lower().endswith('.docx'):
            text = docx2txt.process(filepath)
    except Exception as e:
        print(f"Error extracting text from {filepath}: {e}")
    return text

def extract_skills_from_text(text, all_skills):    # function for extracting skills from text

    extracted_skills = set()
    text_lower = text.lower()
    for skill in all_skills:
        # regular expression for whole word matching
        if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text_lower):
            extracted_skills.add(skill)
    return list(extracted_skills)

def generate_skill_gap_chart(user_skills, required_skills, predicted_title):   # fuction for skills gap visualization

    try:
        import matplotlib
        matplotlib.use('Agg') #  for use without GUI backend 
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is not installed. Please install.")
        return None


    user_skill_set = set(s.lower() for s in user_skills)
    required_skill_set = set(s.lower() for s in required_skills)

    matched_skills_count = len(user_skill_set.intersection(required_skill_set))
    missing_skills_count = len(required_skill_set - user_skill_set)

   # if there is no skills gap then return None
    if not required_skill_set:
        return None

    labels = ['In Your Resume', ' (Missing)']
    sizes = [matched_skills_count, missing_skills_count]
    colors = ['#28a745', '#ffc107'] 
    explode = (0, 0.1) if missing_skills_count > 0 else (0, 0)

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=140, colors=colors, textprops={'fontsize': 14, 'color': 'black'})
    ax1.axis('equal')
    plt.title(f'{predicted_title} Skills Gap', fontsize=16, pad=20)
    
    chart_path = os.path.join(app.config['UPLOAD_FOLDER'], "skill_gap_chart.png")
    plt.savefig(chart_path)
    plt.close()
    return os.path.basename(chart_path)

def get_roadmap_from_gemini(job_title):  # function to generate roadmap using API

    if not GEMINI_API_KEY or not genai:
        return " ERROR: Gemini API is not configure. Please add API key in app.py"
    try:
        model = genai.GenerativeModel('gemini-2.5-pro')

        prompt = f"""
You are a professional career coach. Your task is to create a comprehensive and well-structured career roadmap for someone who wants to become a '{job_title}'.

Instructions:
- Write the entire response in clear and professional English.
- Format the output using clean Markdown with proper headings (#, ##) and bullet points (-).
- Avoid any asterisk-based formatting.
- Ensure the layout is easy to read, logically structured, and consistent throughout.

Roadmap Format:

# Career Roadmap: {job_title}

## Overview
- Provide a brief summary of the role and its importance in the industry.
- Mention key responsibilities and typical career growth path.

## Week 1-4: Foundational Skills
### Objectives
- List 3 to 5 core skills or topics to learn during this phase.
### Learning Tasks
- Briefly describe what the learner should do each week.
### Resources
- Recommend 1-2 quality online resources per topic (YouTube, Udemy, official docs, etc.)

## Week 5-8: Intermediate Skills
### Objectives
- Cover intermediate-level concepts that build upon the basics.
### Practice & Assignments
- Suggest small projects, exercises, or challenges.
### Resources
- Provide recommended learning materials with brief explanations.

## Week 9-12: Advanced Topics
### Objectives
- Introduce advanced subjects relevant to the job role.
### Real-World Applications
- Explain how these skills are applied in industry settings.
### Resources
- Suggest trusted sources to master these topics.

## Week 13-16: Project Phase
### Capstone Projects
- Suggest 1 or 2 meaningful projects to showcase skills.
### Tools & Technologies
- List tools, frameworks, or platforms the learner should use.
### Guidance
- Provide links to example projects or tutorials if available.

## Optional: Bonus Section
- Tips for resume building, certifications, portfolio creation, or interview preparation (only if applicable to the role).

## Final Advice
- End with a short motivational note to encourage consistent learning and self-discipline.

Make sure the roadmap is practical, actionable, and tailored to someone starting this journey from scratch.
"""


        response = model.generate_content(prompt)
        # Markdown to HTML convert 
        return markdown2.markdown(response.text)
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return f"error occur at the time of roadmap generation: {e}. please check your API key."


# --- FLASK ROUTES ---

@app.route('/')
def index():
    """render Home Page."""
    return render_template('index.html')


@app.route('/AboutUs')
def about():
    return render_template('AboutUs.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Resume upload, process, and render result page."""
    if not all([model, tfidf, edu_encoder, loc_encoder, label_encoder, job_roles_skills]):
        return "Server is not configured properly.", 500

    if 'resume' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['resume']
    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # 1. Extract text and skills from resume.
        resume_text = extract_text_from_resume(filepath)
        if not resume_text:
            return "Text could not be extracted. Please check your file.", 400
            
        all_possible_skills = {skill for skills in job_roles_skills.values() for skill in skills}
        user_skills = extract_skills_from_text(resume_text, all_possible_skills)

        # 2. Prepare input for model
        skills_tfidf = tfidf.transform([', '.join(user_skills)]).toarray()
        
        # Default values
        education_level = 'B.Tech'
        location = 'Bangalore'
        experience = 2 # years

        education_encoded = edu_encoder.transform([[education_level]])
        location_encoded = loc_encoder.transform([[location]])
        experience_reshaped = np.array([[experience]])
        
        X_new = np.hstack((skills_tfidf, education_encoded, location_encoded, experience_reshaped))

        # 3. Job title prediction
        prediction_index = model.predict(X_new)[0]
        predicted_job_title = label_encoder.inverse_transform([prediction_index])[0]

        # 4. Required and Missing skills calculate 
        required_skills = job_roles_skills.get(predicted_job_title, [])
        missing_skills = list(set(required_skills) - set(user_skills))
        
        # 5. Skill gap chart generate 
        chart_filename = generate_skill_gap_chart(user_skills, required_skills, predicted_job_title)
        chart_url = url_for('static_file', filename=chart_filename) if chart_filename else None

        # 6. Generate result page with whole data
        return render_template(
            'result.html',
            job_title=predicted_job_title,
            user_skills=user_skills,
            missing_skills=missing_skills,
            chart_url=chart_url
        )

@app.route('/roadmap')
def roadmap():
    """Generate roadmap and render."""
    job_title = request.args.get('job_title', 'Software Developer')
    roadmap_html_content = get_roadmap_from_gemini(job_title)
    return render_template('roadmap.html', job_title=job_title, roadmap=roadmap_html_content)

@app.route('/download_roadmap')
def download_roadmap():
    """Roadmap ko PDF format me download karne ke liye."""
    job_title = request.args.get('job_title', 'Software Developer')
    
    # Roadmap content dobara generate karein
    roadmap_html = get_roadmap_from_gemini(job_title)
    # HTML se saaf text nikalein (PDF ke liye)
    soup = BeautifulSoup(roadmap_html, 'html.parser')
    roadmap_text = soup.get_text()

    pdf = FPDF()
    pdf.add_page()
    

    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, txt=f"{job_title} Roadmap", ln=True, align='C')
    pdf.ln(10)
    
    
    pdf.set_font('Arial', '', 11)
    

    safe_text = roadmap_text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 8, txt=safe_text)
    
    pdf_filename = f'{job_title.replace(" ", "_")}_roadmap.pdf'
    pdf_output_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
    pdf.output(pdf_output_path)

    return send_from_directory(app.config['UPLOAD_FOLDER'], pdf_filename, as_attachment=True)
    
@app.route('/uploads/<filename>')
def static_file(filename):
    """To save file from static folder (like chart image)."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    # to install important libraries
    try:
        import bs4
        import markdown2
        import matplotlib
    except ImportError:
        print("\n--- install important libraries ---")
        print("Some important packages (beautifulsoup4, markdown2, matplotlib) not installed.")
        print("please install them: pip install beautifulsoup4 markdown2 matplotlib\n")

    app.run(debug=True)
