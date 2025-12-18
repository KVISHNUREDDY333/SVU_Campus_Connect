import os
import json
from typing import List, Optional, Union, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv

# 1. Load Environment Variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# 2. Initialize App
app = FastAPI(title="SV University Campus Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Data Management
DATA_FILE = "data.json"

def load_data():
    if not os.path.exists(DATA_FILE):
        return {"faqs": []} # Return default structure
    with open(DATA_FILE, "r") as f:
        try:
            content = json.load(f)
            if isinstance(content, list):
                return {"faqs": content} # Backward compatibility
            return content
        except json.JSONDecodeError:
            return {"faqs": []}

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

# 4. RAG Logic: find_relevant_context
def find_relevant_context(user_query: str):
    data = load_data()
    searchable_docs = []
    
    # 1. FAQs
    if "faqs" in data and isinstance(data["faqs"], list):
        searchable_docs.extend(data["faqs"])
        
    # 2. Facilities
    if "facilities" in data and isinstance(data["facilities"], list):
        for item in data["facilities"]:
            searchable_docs.append({
                "question": f"Facility: {item.get('name')} - {item.get('location')}",
                "answer": f"{item.get('name')} is located at {item.get('location')}. {item.get('description')}"
            })
            
    # 3. Academic Programs
    if "academic_programs" in data and isinstance(data["academic_programs"], list):
        for item in data["academic_programs"]:
            searchable_docs.append({
                "question": f"Program: {item.get('name')}",
                "answer": f"{item.get('name')}. {item.get('description')} Fee: {item.get('fee')}"
            })

    # 4. Placements
    if "placements" in data and isinstance(data["placements"], dict):
        p = data["placements"]
        searchable_docs.append({
            "question": "Placement details and top recruiters",
            "answer": f"{p.get('summary')} Top Recruiters: {', '.join(p.get('top_recruiters', []))}"
        })
        
    # 5. Syllabi
    if "syllabi" in data and isinstance(data["syllabi"], dict):
        searchable_docs.append({
            "question": "Syllabus information",
            "answer": data["syllabi"].get("summary", "")
        })

    user_tokens = user_query.lower().split()
    relevant_chunks = []
    
    # Simple keyword matching algorithm
    for entry in searchable_docs:
        # Ensure keys exist
        q_text = entry.get('question', '')
        a_text = entry.get('answer', '')
        cat_text = entry.get('category', '')
        
        # Boost matches in Question
        text_to_search = (q_text + " " + a_text + " " + cat_text).lower()
        
        score = 0
        for token in user_tokens:
            if token in text_to_search:
                score += 1
            if token in q_text.lower(): # Bonus for matching question directly
                score += 1
        
        if score > 0:
            # Add category prefix to the chunk for better LLM context
            prefix = f"[{cat_text}] " if cat_text else ""
            relevant_chunks.append((score, f"{prefix}Q: {q_text}\nA: {a_text}"))
    
    # Sort by score (descending) and return top 5
    relevant_chunks.sort(key=lambda x: x[0], reverse=True)
    top_matches = [chunk[1] for chunk in relevant_chunks[:5]]
    
    return "\n\n".join(top_matches)

# 5. Gemini AI Configuration
# Using verified working model alias
model = genai.GenerativeModel('models/gemini-2.5-flash-lite') 

# --- Dynamic Web Search & Scraping ---
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus

def search_university_website(query: str):
    """
    Searches svuniversity.edu.in. 
    1. Checks for specific keywords to scrape known pages directly (Notifications, Exams).
    2. Falls back to Google Search for other queries.
    """
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    extracted_content = ""
    
    # --- Strategy 1: Direct Page Scraping based on Keywords ---
    direct_urls = []
    lower_query = query.lower()
    
    if any(k in lower_query for k in ['notification', 'latest', 'update', 'news']):
        direct_urls.append("https://svuniversity.edu.in/notifications/")
        
    if any(k in lower_query for k in ['exam', 'result', 'schedule', 'time table', 'circular']):
        direct_urls.append("https://svuniversity.edu.in/exams-circulars/")
        
    for url in direct_urls:
        try:
            print(f"Direct scraping: {url}")
            res = requests.get(url, headers=headers, timeout=10, verify=False)
            if res.status_code == 200:
                soup = BeautifulSoup(res.text, 'html.parser')
                
                # Extract links/headlines from these pages.
                # Usually these pages have lists of links.
                # We'll try to find decent text chunks or lists.
                
                # Assumption: Notifications are often in tables or lists (ul/li) or div rows.
                # We grab text from the main content area.
                main_content = soup.find('main') or soup.find(class_='content') or soup.body
                
                # Extract text from the first 15 list items or paragraphs
                items = main_content.find_all(['li', 'p', 'tr'])
                text_chunk = []
                for item in items[:20]: # First 20 items
                    t = item.get_text().strip()
                    if t and len(t) > 10: # meaningful text
                        text_chunk.append(t)
                
                if text_chunk:
                   extracted_content += f"\n**Source:** {url}\n**Relevant Updates:**\n" + "\n- ".join(text_chunk) + "\n\n"
        except Exception as e:
            print(f"Error direct scraping {url}: {e}")

    if extracted_content:
        return extracted_content

    # --- Strategy 2: Fallback to Search ---
    try:
        search_query = f"site:svuniversity.edu.in {query}"
        url = f"https://www.google.com/search?q={quote_plus(search_query)}"
        
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        
        links = []
        # Try multiple selectors for Google results
        for selector in ['div.yuRUbf a', 'div.g a', 'a']:
            found = soup.select(selector)
            for result in found:
                href = result.get('href')
                if href and 'svuniversity.edu.in' in href and not 'google.com' in href:
                    if href not in links:
                        links.append(href)
            if links: break
        
        links = links[:2]
        if not links:
            return None

        print(f"Found fallback links: {links}")
        
        for link in links:
            try:
                page_res = requests.get(link, headers=headers, timeout=5, verify=False)
                page_soup = BeautifulSoup(page_res.text, 'html.parser')
                paragraphs = page_soup.find_all('p')
                text = " ".join([p.get_text() for p in paragraphs[:8]])
                extracted_content += f"\nSource: {link}\nContent: {text[:800]}...\n"
            except Exception as e:
                print(f"Error scraping {link}: {e}")
                
        return extracted_content if extracted_content else None
        
    except Exception as e:
        print(f"Web Search Error: {e}")
        return None

# --- API Endpoints ---

class ChatRequest(BaseModel):
    message: str

class FAQItem(BaseModel):
    id: Optional[Union[str, int]] = None
    question: str
    answer: str
    category: Optional[str] = "General"

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    user_query = request.message
    
    # Step 1: Check Local Context (Fast & Reliable)
    local_context = find_relevant_context(user_query)
    
    # Step 2: Dynamic Web Search (If local context is weak or user asks for specific live info)
    # We always fetch web content if the answer isn't obvious, to ensure freshness.
    # Optimization: Only search web if local context score is low or query implies "news/latest"
    web_context = ""
    if "latest" in user_query.lower() or "news" in user_query.lower() or len(local_context) < 50:
        print("Performing live web search...")
        web_context = search_university_website(user_query)
    
    combined_context = ""
    if local_context:
        combined_context += f"**Local Database Info:**\n{local_context}\n\n"
    if web_context:
        combined_context += f"**Live Website Info:**\n{web_context}\n\n"
    
    if not combined_context:
        combined_context = "No relevant information found in local database or on the official website."

    # Step 3: System Prompt
    system_instruction = """
    You are 'CampusConnect AI', a helpful assistant for Sri Venkateswara University. 
    You have access to both a local database and real-time information scraped from the official website.
    Always prioritize the **Live Website Info** if it appears more recent or relevant.
    
    Answer the student's question clearly and professionally.
    If the information is from the website, cite the source URL provided in the context.
    If the answer is truly not found in the context, strictly state:
    "I'm sorry, I couldn't find that specific information on the SVU website or in my database. Please check https://svuniversity.edu.in/ directly."
    Do not make up facts.
    """
    
    prompt = f"{system_instruction}\n\nContext:\n{combined_context}\n\nUser Question: {user_query}"

    # Step 4: Generate Response
    try:
        response = model.generate_content(prompt)
        return {"response": response.text}
    except Exception as e_flash:
        print(f"Gemini Flash Error: {e_flash}")
        try:
            print("Attempting fallback to gemini-pro-latest...")
            fallback_model = genai.GenerativeModel('gemini-pro-latest')
            response = fallback_model.generate_content(prompt)
            return {"response": response.text}
        except Exception as e_pro:
            print(f"Fallback Error: {e_pro}")
            return {"response": f"**Network Unavailable**\n\nI tried to search the web but couldn't connect. Here is what I found locally:\n\n{local_context}"}

# --- Admin CRUD Endpoints ---

@app.get("/api/faqs")
def get_faqs():
    data = load_data()
    return data.get("faqs", [])

@app.post("/api/faqs")
def add_faq(faq: FAQItem):
    data = load_data()
    if "faqs" not in data:
        data["faqs"] = []
    
    # Simple ID generation
    new_id = f"faq-{len(data['faqs']) + 100}" # Avoid collisions with existing
    new_entry = {
        "id": new_id, 
        "question": faq.question, 
        "answer": faq.answer,
        "category": faq.category or "General"
    }
    data["faqs"].append(new_entry)
    save_data(data)
    return {"message": "Success", "id": new_id}

@app.delete("/api/faqs/{faq_id}")
def delete_faq(faq_id: str):
    data = load_data()
    if "faqs" in data:
        # Filter (handling both str and int IDs from legacy)
        data["faqs"] = [d for d in data["faqs"] if str(d.get('id')) != str(faq_id)]
        save_data(data)
    return {"message": "Deleted"}

@app.put("/api/faqs/{faq_id}")
def update_faq(faq_id: str, faq: FAQItem):
    data = load_data()
    if "faqs" in data:
        for item in data["faqs"]:
            if str(item.get('id')) == str(faq_id):
                item['question'] = faq.question
                item['answer'] = faq.answer
                item['category'] = faq.category or item.get('category', 'General')
                save_data(data)
                return {"message": "Updated"}
    raise HTTPException(status_code=404, detail="Not found")

# Serve Frontend with cache busting
app.mount("/", StaticFiles(directory="static", html=True), name="static")

@app.middleware("http")
async def add_no_cache_header(request, call_next):
    response = await call_next(request)
    if request.url.path in ["/script.js", "/style.css"]:
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8500)