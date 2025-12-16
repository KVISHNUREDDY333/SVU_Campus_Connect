import requests
from bs4 import BeautifulSoup
import json
import os
import time
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Error: GOOGLE_API_KEY not found in .env file.")
    exit()

genai.configure(api_key=api_key)

# Configuration
DATA_FILE = "data.json"
MODEL_NAME = "gemini-1.5-flash" # Standard efficient model
# Fallback model if flash fails
FALLBACK_MODEL = "gemini-pro"

urls_to_scrape = [
    # HOME & BASIC PAGES
    "https://svuniversity.edu.in/",
    "https://svuniversity.edu.in/about/",
    
    # LEADERSHIP & ADMINISTRATION
    "https://svuniversity.edu.in/vice-chancellor/",
    "https://svuniversity.edu.in/rector/",
    "https://svuniversity.edu.in/registrar/",
    "https://svuniversity.edu.in/former-vice-chancellors/",
    "https://svuniversity.edu.in/former-rectors/",
    "https://svuniversity.edu.in/former-registars/",
    "https://svuniversity.edu.in/former-joint-registrars/",
    "https://svuniversity.edu.in/executive-council/",
    "https://svuniversity.edu.in/academic-senate/",
    "https://svuniversity.edu.in/finance-committee/",
    "https://svuniversity.edu.in/officers/",
    "https://svuniversity.edu.in/urc-members",
    "https://svuniversity.edu.in/deputy-registrars",
    "https://svuniversity.edu.in/aao",
    "https://svuniversity.edu.in/cpc-members",
    
    # COLLEGES & DEPARTMENTS
    "https://svuniversity.edu.in/college-of-arts/",
    "https://svuniversity.edu.in/college-principal-cfa/",
    "https://svuniversity.edu.in/sv-arts-department/",
    "https://svuniversity.edu.in/college-events-cfa/",
    "https://svuniversity.edu.in/college-staff-cfa/",
    
    "https://svuniversity.edu.in/college-of-science/",
    "https://svuniversity.edu.in/college-of-science-principal/",
    "https://svuniversity.edu.in/college-of-science-department/",
    "https://svuniversity.edu.in/events-achievements-cfs/",
    "https://svuniversity.edu.in/collage-of-science-administration/",
    
    "https://svuniversity.edu.in/college-of-engineering/",
    "https://svuniversity.edu.in/college-departments-cfe/",
    "https://svuniversity.edu.in/college-events-cfe/",
    "https://svuniversity.edu.in/collage-of-engineering-administration-staff/",
    
    "https://svuniversity.edu.in/about-cmcs/",
    "https://svuniversity.edu.in/collage-of-commerce/",
    "https://svuniversity.edu.in/college-departments-cmcs/",
    "https://svuniversity.edu.in/college-events-cm-cs/",
    "https://svuniversity.edu.in/collage-of-commerce-cmcs/",
    
    "https://svuniversity.edu.in/college-of-pharmaceutical-sciences/",
    "https://svuniversity.edu.in/collage-of-pharmacy/",
    "https://svuniversity.edu.in/college-departments-cfp/",
    "https://svuniversity.edu.in/college-events-cfp/",
    "https://svuniversity.edu.in/collage-of-pharmacy-administration/",
    "https://svuniversity.edu.in/collage-of-pharmacy-sif-report/",
    
    # CENTRES & FACILITIES
    "https://svuniversity.edu.in/advanced-centre-for-atmospheric-sciences",
    "https://svuniversity.edu.in/bioinformatics-infrastructure-facility-bif",
    "https://svuniversity.edu.in/cseap-studies-center",
    "https://svuniversity.edu.in/computer-center/",
    "https://svuniversity.edu.in/cerdat/",
    "https://svuniversity.edu.in/doa/",
    "https://svuniversity.edu.in/dst-purse-centre-3",
    "https://svuniversity.edu.in/mmttc-center/",
    "https://svuniversity.edu.in/ori-center",
    "https://svuniversity.edu.in/usi-center",
    
    # ACADEMICS
    "https://svuniversity.edu.in/academic-programme/",
    "https://svuniversity.edu.in/professional-courses/",
    "https://svuniversity.edu.in/degree-course-syllabus/",
    "https://svuniversity.edu.in/pg-course-syllabus/",
    
    # DEANS
    "https://svuniversity.edu.in/dean-development",
    "https://svuniversity.edu.in/dean-rd",
    "https://svuniversity.edu.in/dean-commerce-management/",
    "https://svuniversity.edu.in/dean-cdc",
    "https://svuniversity.edu.in/dean-international-relations",
    "https://svuniversity.edu.in/dean-it",
    "https://svuniversity.edu.in/dean-faculty-of-sciences",
    "https://svuniversity.edu.in/dean-of-examinations/",
    
    # RESEARCH & AFFILIATION
    "https://svuniversity.edu.in/research/",
    "https://svuniversity.edu.in/college-affiliation/",
    "https://svuniversity.edu.in/control-of-examination/",
    
    # CAMPUS FACILITIES & SERVICES
    "https://svuniversity.edu.in/facilities/",
    "https://svuniversity.edu.in/library/",
    "https://svuniversity.edu.in/stadium/",
    "https://svuniversity.edu.in/health-center/",
    "https://svuniversity.edu.in/womens-hostels/",
    "https://svuniversity.edu.in/svu-guest-house/",
    "https://svuniversity.edu.in/campus-school/",
    "https://svuniversity.edu.in/annyamaya-bhavan/",
    "https://svuniversity.edu.in/internet-facility/",
    "https://svuniversity.edu.in/s-v-university-post-office/",
    "https://svuniversity.edu.in/labs/",
    "https://svuniversity.edu.in/rs-hostel",
    "https://svuniversity.edu.in/open-air-theatre/",
    
    # STUDENT SERVICES & ACTIVITIES
    "https://svuniversity.edu.in/nss/",
    "https://svuniversity.edu.in/ncc",
    "https://svuniversity.edu.in/day-care-centre/",
    "https://svuniversity.edu.in/sbi-svu-campus-branch/",
    "https://svuniversity.edu.in/gallery/",
    "https://svuniversity.edu.in/sports-games/",
    
    # QUALITY ASSURANCE
    "https://svuniversity.edu.in/iqac/",
    "https://svuniversity.edu.in/naac",
    "https://svuniversity.edu.in/iiqa/",
    "https://svuniversity.edu.in/ssr",
    "https://svuniversity.edu.in/dvv/"
]

def get_page_text(url):
    try:
        # Verify=False for SVU website certificate issues
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, verify=False, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove scripts and styles
        for script in soup(["script", "style", "nav", "footer"]):
            script.decompose()
            
        # Get text
        text = soup.get_text(separator=' ', strip=True)
        # Clean up whitespace
        text = ' '.join(text.split())
        return text[:10000] # Limit context window
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return None

def classify_category(question, answer):
    text = (question + " " + answer).lower()
    if any(k in text for k in ["vice-chancellor", "rector", "registrar", "dean", "principal", "executive council", "senate", "officer", "administration", "leadership", "governance"]):
        return "Administration"
    if any(k in text for k in ["admission", "apply", "fee", "scholarship", "eligibility", "entrance", "exam", "rank", "seat", "application", "dates"]):
        return "Admissions"
    if any(k in text for k in ["course", "program", "syllabus", "curriculum", "degree", "b.tech", "m.tech", "phd", "academic", "department", "faculty", "class", "studies"]):
        return "Academics"
    if any(k in text for k in ["hostel", "library", "wifi", "transport", "bus", "sports", "gym", "canteen", "lab", "facility", "infrastructure", "building", "campus", "center"]):
        return "Facilities"
    if any(k in text for k in ["address", "phone", "email", "location", "where is", "contact", "reach"]):
        return "Contact & Location"
    if any(k in text for k in ["placement", "job", "recruit", "salary", "package", "internship", "career"]):
        return "Placements"
    if any(k in text for k in ["history", "established", "motto", "vision", "mission", "naac", "ranking", "about", "founder", "accreditation"]):
        return "General Info"
    return "Other"

def generate_faqs(text, url):
    model = genai.GenerativeModel(MODEL_NAME)
    prompt = f"""
    Analyze the following text scraped from {url} of Sri Venkateswara University.
    Extract key facts, dates, names, roles, and procedures.
    Generate 5 to 10 high-quality Question-Answer pairs (FAQs) based ONLY on this text.
    Format the output as a JSON array of objects, where each object has "question" and "answer" keys.
    Do not add any markdown formatting like ```json. just return the raw JSON array.
    
    Text:
    {text}
    """
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            content = response.text.strip()
            # Clean markdown if present
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
                
            faqs = json.loads(content)
            # Add categorization immediately
            for faq in faqs:
                faq['category'] = classify_category(faq.get('question',''), faq.get('answer',''))
            return faqs
            
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                wait_time = (attempt + 1) * 10
                print(f"  !! Rate limit hit. Waiting {wait_time}s before retry {attempt+1}/{max_retries}...")
                time.sleep(wait_time)
            else:
                print(f"Error generating FAQS for {url}: {e}")
                return []
    
    print(f"  !! Failed to generate FAQs for {url} after {max_retries} retries.")
    return []

def update_database(new_faqs):
    if not os.path.exists(DATA_FILE):
        data = {"faqs": []}
    else:
        with open(DATA_FILE, 'r') as f:
            try:
                data = json.load(f)
            except:
                data = {"faqs": []}
    
    if "faqs" not in data:
        data["faqs"] = []
    
    # Check for duplicates based on question
    existing_questions = {ftp['question'].lower() for ftp in data["faqs"] if 'question' in ftp}
    
    count = len(data["faqs"])
    added_count = 0
    for faq in new_faqs:
        if faq['question'].lower() in existing_questions:
            continue
            
        count += 1
        faq["id"] = f"auto-gen-{int(time.time())}-{count}" # More unique ID
        data["faqs"].append(faq)
        added_count += 1
        
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Updates saved to {DATA_FILE} (+{added_count} new)")

def main():
    # Load existing processed URLs
    processed_urls = set()
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r') as f:
                data = json.load(f)
                if "faqs" in data:
                    for item in data["faqs"]:
                        if "source" in item:
                            processed_urls.add(item["source"])
            print(f"Loaded {len(processed_urls)} already processed URLs.")
        except Exception as e:
            print(f"Error loading existing data: {e}")

    print(f"Starting scraping of {len(urls_to_scrape)} URLs...")
    
    batch_faqs = []
    total_generated = 0
    
    for i, url in enumerate(urls_to_scrape):
        if url in processed_urls:
            print(f"[{i+1}/{len(urls_to_scrape)}] Skipping {url} (Already processed)")
            continue

        print(f"[{i+1}/{len(urls_to_scrape)}] Processing {url}...")
        text = get_page_text(url)
        if text:
            print(f"  - Content extracted ({len(text)} chars). Generating FAQs...")
            faqs = generate_faqs(text, url)
            if faqs:
                print(f"  - Generated {len(faqs)} FAQs.")
                for faq in faqs:
                   faq['source'] = url
                batch_faqs.extend(faqs)
                total_generated += len(faqs)
            else:
                print("  - No FAQs generated.")
            
            # Rate limiting
            time.sleep(2) 
        else:
            print("  - Skipping.")

        # Save every 5 URLs or if it's the last one
        if len(batch_faqs) > 0 and ((i + 1) % 5 == 0 or (i + 1) == len(urls_to_scrape)):
             print(f"  >> Saving batch of {len(batch_faqs)} FAQs to database...")
             update_database(batch_faqs)
             batch_faqs = [] # Reset batch

    print(f"Completed! Total new FAQs generated: {total_generated}")

if __name__ == "__main__":
    # Suppress SSL warnings
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    main()
