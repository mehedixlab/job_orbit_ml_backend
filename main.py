from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import requests
import PyPDF2
import io
import re
from sentence_transformers import SentenceTransformer, util

# FastAPI অ্যাপ ইনিশিয়ালাইজ করা
app = FastAPI(title="JobOrbitBD Advanced ML API")

# NLP Semantic Embedding মডেল লোড করা হচ্ছে
print("Loading Advanced NLP Embedding Model (MiniLM)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
# (all_-MiniLM-L6-v2) Sentence Transformer Model for semantic skill matching
print("Model Loaded Successfully!")

# ডাটা রিসিভ করার মডেল 
class MatchRequest(BaseModel):
    student_skills: str
    job_requirements: str
    job_description: Optional[str] = ""
    cv_url: Optional[str] = None

#  1. PDF Text Extraction
def extract_text_from_pdf_url(pdf_url):
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        pdf_file = io.BytesIO(response.content)
        reader = PyPDF2.PdfReader(pdf_file)
        extracted_text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return extracted_text.strip()
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

# 2. Experience Matching (Advanced Regex)
def extract_experience(text):
    if not text:
        return 0
    text = text.lower()
    try:
        m1 = re.search(r'(\d+)\s*\+?\s*(?:year|yr)s?\s*(?:of\s*)?exp', text)
        if m1: return int(m1.group(1))
        
        m2 = re.search(r'exp[a-z]*\s*(?::|-|is)?\s*(\d+)\s*\+?\s*(?:year|yr)s?', text)
        if m2: return int(m2.group(1))
        
        if "exp" in text:
            m3 = re.search(r'(\d+)\s*\+?\s*(?:year|yr)s?', text)
            if m3: return int(m3.group(1))
    except Exception as e:
        print(f"Experience parsing error: {e}")
    return 0

# 3. Skill Weighting (Required vs Preferred)
def classify_skills(job_req_string):
    required_skills = []
    preferred_skills = []
    # শুধুমাত্র কমা দিয়ে আলাদা করা স্কিলগুলো নিবে
    skills = [s.strip() for s in job_req_string.split(',') if s.strip()]
    
    for skill in skills:
        if re.search(r'(?i)(preferred|plus|nice to have)', skill):
            clean_skill = re.sub(r'(?i)\(?(preferred|plus|nice to have)\)?', '', skill).strip()
            preferred_skills.append(clean_skill if clean_skill else skill)
        else:
            required_skills.append(skill)
            
    return required_skills, preferred_skills

@app.get("/")
def read_root():
    return {"message": "Welcome to JobOrbitBD Advanced AI Matching API 🚀"}

@app.post("/calculate-match")
def calculate_match(data: MatchRequest):
    # ১. স্টুডেন্টের প্রোফাইল তৈরি
    cv_text = ""
    if data.cv_url:
        cv_text = extract_text_from_pdf_url(data.cv_url)
        
    full_student_profile = data.student_skills + " " + cv_text
    student_skills_list = [s.strip() for s in data.student_skills.split(',') if s.strip()]

    # ২. স্কিল ক্লাসিফাই করার জন্য শুধুমাত্র "job_requirements" ফিল্ড ব্যবহার করা হচ্ছে
    required_skills, preferred_skills = classify_skills(data.job_requirements)
    
    # ৩. এক্সপেরিয়েন্স খোঁজার জন্য ডেসক্রিপশন এবং স্কিলস দুটোই একসাথে মিলিয়ে চেক করা হচ্ছে
    full_job_text = data.job_requirements + " " + data.job_description
    job_exp = extract_experience(full_job_text)
    
    student_exp = extract_experience(full_student_profile)

    # ৪. Experience Match ক্যালকুলেশন (ওয়েট: 20%)
    exp_match_score = 100.0
    if job_exp > 0:
        if student_exp >= job_exp:
            exp_match_score = 100.0
        else:
            exp_match_score = (student_exp / job_exp) * 100.0
    elif job_exp == 0 and student_exp == 0:
        exp_match_score = 100.0

    # ৫. Semantic Similarity & Skill Match (ওয়েট: 80%)
    matched_required = []
    missing_required = []
    matched_preferred = []
    
    def check_skill_match(target_skill):
        if target_skill.lower() in cv_text.lower() or any(target_skill.lower() in s.lower() for s in student_skills_list):
            return True
        
        if student_skills_list:
            skill_emb = model.encode(target_skill, convert_to_tensor=True)
            student_embs = model.encode(student_skills_list, convert_to_tensor=True)
            sims = util.pytorch_cos_sim(skill_emb, student_embs)[0]
            if sims.max().item() > 0.60:
                return True
        return False

    for req in required_skills:
        if check_skill_match(req):
            matched_required.append(req)
        else:
            missing_required.append(req)

    for pref in preferred_skills:
        if check_skill_match(pref):
            matched_preferred.append(pref)

    total_skills = len(required_skills) + len(preferred_skills)
    total_matched = len(matched_required) + len(matched_preferred)
    skill_match_score = (total_matched / total_skills) * 100.0 if total_skills > 0 else 100.0

    if len(required_skills) > 0:
        req_score = (len(matched_required) / len(required_skills)) * 100
        skill_match_score = (req_score * 0.8) + (skill_match_score * 0.2)

    # ৬. Overall Job Matching Score
    overall_match = (skill_match_score * 0.8) + (exp_match_score * 0.2)
    
    overall_match = round(overall_match, 1)
    skill_match_score = round(skill_match_score, 1)
    exp_match_score = round(exp_match_score, 1)

    breakdown_msg = (
        f"Overall Match: {overall_match}%\n\n"
        f"Skills Match: {skill_match_score}%\n"
        f"Experience Match: {exp_match_score}%\n"
        f"Required Skills: {len(matched_required)}/{len(required_skills)}\n"
        f"Preferred Skills: {len(matched_preferred)}/{len(preferred_skills)}"
    )

    return {
        "match_score": f"{overall_match}%",
        "missing_skills": missing_required, # শুধু আসল মিসিং স্কিলসগুলোই এখানে আসবে
        "message": breakdown_msg
    }