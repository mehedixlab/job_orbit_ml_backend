from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import requests
import PyPDF2
import io
import re
import os

# FastAPI অ্যাপ ইনিশিয়ালাইজ করা
app = FastAPI(title="JobOrbitBD Advanced ML API")

HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# --- Request Models ---
class MatchRequest(BaseModel):
    student_skills: Optional[str] = "" 
    job_requirements: str
    job_description: Optional[str] = ""
    cv_url: Optional[str] = None

class ExtractRequest(BaseModel):
    cv_url: str

class AIHubRequest(BaseModel):
    action: str
    skills: str
    target_job: Optional[str] = ""
    answer: Optional[str] = ""
    query: Optional[str] = ""

# 🌟 1. FIX: User-Agent যুক্ত করে PDF Text Extraction 🌟
def extract_text_from_pdf_url(pdf_url):
    try:
        # User-Agent না দিলে Cloudinary/Firebase অনেক সময় পাইথনকে ব্লক করে দেয়
        req_headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(pdf_url, headers=req_headers, timeout=10)
        response.raise_for_status()
        
        pdf_file = io.BytesIO(response.content)
        reader = PyPDF2.PdfReader(pdf_file)
        
        extracted_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + " \n "
                
        # অতিরিক্ত স্পেস মুছে ফেলা
        clean_text = re.sub(r'\s+', ' ', extracted_text)
        return clean_text.strip()
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

# 🌟 2. FIX: Aggressive Skill Extraction 🌟
def extract_skills_from_cv_text(text):
    if not text: return []
    
    extracted_skills = set()
    text_lower = text.lower()

    # --- পদ্ধতি ১: Broad Dictionary (১০০% ক্যাচ করবে) ---
    common_tech_skills = [
        "flutter", "dart", "python", "django", "fastapi", "rest api", "http api", "rest", "http",
        "machine learning", "ml", "artificial intelligence", "ai", "react", "react native",
        "javascript", "java", "c++", "c#", "html", "css", "sql", "mysql", "postgresql",
        "mongodb", "firebase", "git", "github", "docker", "aws", "gcp", "azure",
        "ui/ux", "figma", "api development", "data science", "nlp", "deep learning"
    ]
    
    for skill in common_tech_skills:
        if skill in text_lower:
            if len(skill) <= 3:
                # ছোট শব্দ যেন অন্য শব্দের অংশ না হয় (যেমন: hTML এর ml)
                if re.search(rf'\b{re.escape(skill)}\b', text_lower):
                    if skill == "ml": extracted_skills.add("Machine Learning")
                    elif skill == "ai": extracted_skills.add("Artificial Intelligence")
                    elif skill == "api": extracted_skills.add("API Development")
                    else: extracted_skills.add(skill.upper())
            else:
                extracted_skills.add(skill.title())

    # --- পদ্ধতি ২: Regex Fallback (হেডিং থেকে বের করা) ---
    try:
        heading_pattern = r'(?i)(?:technical\s+|core\s+)?skills?(?:\s+highlights|\s+summary)?\s*[:\-]*\s*'
        next_section_pattern = r'(?i)(?:EXPERIENCE|EDUCATION|PROJECTS|CERTIFICATIONS|WORK HISTORY|EMPLOYMENT|SUMMARY|OBJECTIVE|LANGUAGES|REFERENCES)'
        
        match = re.search(f'{heading_pattern}(.*?)(?:{next_section_pattern}|$)', text, re.DOTALL)
        if match:
            skills_section = match.group(1)
            raw_skills = re.split(r'[,|\n•\-*]', skills_section)
            for s in raw_skills:
                clean_s = re.sub(r'[^a-zA-Z0-9\s+#.]', '', s).strip()
                if 1 < len(clean_s) < 35:
                    extracted_skills.add(clean_s.title())
    except Exception as e:
        print(f"Regex error: {e}")

    return list(extracted_skills)

# 3. Experience Matching
def extract_experience(text):
    if not text: return 0
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

# 4. Skill Weighting
def classify_skills(job_req_string):
    required_skills = []
    preferred_skills = []
    skills = [s.strip() for s in job_req_string.split(',') if s.strip()]
    for skill in skills:
        if re.search(r'(?i)(preferred|plus|nice to have)', skill):
            clean_skill = re.sub(r'(?i)\(?(preferred|plus|nice to have)\)?', '', skill).strip()
            preferred_skills.append(clean_skill if clean_skill else skill)
        else:
            required_skills.append(skill)
    return required_skills, preferred_skills


# --- API ENDPOINTS ---

@app.get("/")
def read_root():
    return {"message": "Welcome to JobOrbitBD Advanced AI Matching API 🚀"}

@app.post("/extract-cv-skills")
def extract_cv_skills(data: ExtractRequest):
    cv_text = extract_text_from_pdf_url(data.cv_url)
    extracted_skills = extract_skills_from_cv_text(cv_text)
    return {"extracted_skills": extracted_skills}

@app.post("/calculate-match")
def calculate_match(data: MatchRequest):
    cv_text = ""
    if data.cv_url:
        cv_text = extract_text_from_pdf_url(data.cv_url)
        
    student_skills_list = extract_skills_from_cv_text(cv_text)
    
    if not student_skills_list and data.student_skills:
        student_skills_list = [s.strip() for s in data.student_skills.split(',') if s.strip()]

    full_student_profile = " ".join(student_skills_list) + " " + cv_text
    
    required_skills, preferred_skills = classify_skills(data.job_requirements)
    
    full_job_text = data.job_requirements + " " + data.job_description
    job_exp = extract_experience(full_job_text)
    student_exp = extract_experience(full_student_profile)

    exp_match_score = 100.0
    if job_exp > 0:
        if student_exp >= job_exp: exp_match_score = 100.0
        else: exp_match_score = (student_exp / job_exp) * 100.0
    elif job_exp == 0 and student_exp == 0:
        exp_match_score = 100.0

    matched_required = []
    missing_required = []
    matched_preferred = []
    
    # 🌟 5. FIX: Smart Alias Matching Logic 🌟
    def check_skill_match(target_skill):
        target_lower = target_skill.lower()
        cv_lower = cv_text.lower()
        
        # 5.1 Direct Text Match
        if target_lower in cv_lower:
            return True
        for s in student_skills_list:
            if target_lower in s.lower() or s.lower() in target_lower:
                return True

        # 5.2 Smart Alias Group Match
        alias_groups = [
            ["rest", "http", "api"],
            ["machine learning", "ml", "deep learning"],
            ["ui/ux", "user interface", "user experience", "figma", "product design"],
            ["frontend", "front-end", "front end", "react", "flutter"],
            ["backend", "back-end", "back end", "django", "fastapi"]
        ]
        
        for group in alias_groups:
            if any(alias in target_lower for alias in group):
                if any(alias in cv_lower for alias in group):
                    return True
                for s in student_skills_list:
                    if any(alias in s.lower() for alias in group):
                        return True

        # 5.3 Hugging Face API Fallback (Semantic Match)
        if student_skills_list:
            try:
                payload = {
                    "inputs": {"source_sentence": target_skill, "sentences": student_skills_list},
                    "options": {"wait_for_model": True}
                }
                response = requests.post(API_URL, headers=headers, json=payload, timeout=15)
                if response.status_code == 200:
                    scores = response.json()
                    if isinstance(scores, list) and len(scores) > 0:
                        if max(scores) > 0.35: 
                            return True
            except Exception as e:
                print(f"HF Error: {e}")

        return False

    # Check Required & Preferred Skills
    for req in required_skills:
        if check_skill_match(req): matched_required.append(req)
        else: missing_required.append(req)

    for pref in preferred_skills:
        if check_skill_match(pref): matched_preferred.append(pref)

    total_skills = len(required_skills) + len(preferred_skills)
    total_matched = len(matched_required) + len(matched_preferred)
    skill_match_score = (total_matched / total_skills) * 100.0 if total_skills > 0 else 100.0

    if len(required_skills) > 0:
        req_score = (len(matched_required) / len(required_skills)) * 100
        skill_match_score = (req_score * 0.8) + (skill_match_score * 0.2)

    overall_match = (skill_match_score * 0.8) + (exp_match_score * 0.2)

    breakdown_msg = (
        f"Overall Match: {round(overall_match, 1)}%\n\n"
        f"Skills Match: {round(skill_match_score, 1)}%\n"
        f"Experience Match: {round(exp_match_score, 1)}%\n"
        f"Required Skills: {len(matched_required)}/{len(required_skills)}\n"
        f"Preferred Skills: {len(matched_preferred)}/{len(preferred_skills)}"
    )

    return {
        "match_score": f"{round(overall_match, 1)}%",
        "missing_skills": missing_required,
        "message": breakdown_msg
    }

# --- 6. AI Hub Features (Groq API Integration) ---
@app.post("/ai-hub")
def ai_hub_features(data: AIHubRequest):
    raw_key = os.getenv("GROQ_API_KEY")
    
    if not raw_key:
        return {"success": False, "error": "Render Server Error: GROQ_API_KEY is not set in Environment Variables!"}
        
    GROQ_API_KEY = raw_key.strip()
    
    prompt = ""
    if data.action == "generate_questions":
        prompt = (f"You are an expert technical recruiter. The candidate is applying for the '{data.target_job}' role "
                  f"and has the following skills: {data.skills}. Generate exactly 5 technical and 3 behavioral interview "
                  f"questions to test their expertise. Provide only the questions in a clean, numbered format without answers.")
    
    elif data.action == "mock_eval":
        prompt = (f"You are an expert technical interviewer. Evaluate this interview answer provided by a candidate: "
                  f"'{data.answer}'. Provide a score out of 10 and a brief 2-3 sentence constructive feedback on how they can improve.")
                  
    elif data.action == "career_chat":
        prompt = (f"You are an expert career counselor. The user is a student with skills in {data.skills}. "
                  f"Answer their career-related query directly and professionally: '{data.query}'")
        
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # 🌟 চূড়ান্ত ফিক্স: Groq-এর সবচেয়ে স্টেবল এবং ইউনিভার্সাল মডেল (Mixtral) 🌟
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "openai/gpt-oss-20b", 
        "messages": [{"role": "user", "content": prompt}]
    }
    
    try:
        res = requests.post(url, headers=headers, json=payload, timeout=20)
        res.raise_for_status() 
        
        response_data = res.json()
        ai_response = response_data['choices'][0]['message']['content']
        
        return {"success": True, "data": ai_response}
        
    except requests.exceptions.HTTPError as err:
        error_details = res.text
        print(f"Groq HTTP Error: {error_details}")
        return {"success": False, "error": f"API Error: {res.status_code} - {error_details}"}
        
    except Exception as e:
        print(f"Internal Server Error: {str(e)}")
        return {"success": False, "error": f"Server Error: {str(e)}"}