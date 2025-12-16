import argparse
import requests
from bs4 import BeautifulSoup
import json
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Error: GOOGLE_API_KEY not found in .env file.")
    exit(1)

genai.configure(api_key=api_key)

DATA_FILE = "data.json"
MODEL_NAME = "gemini-1.5-flash"

def get_page_text(url):
    try:
        # Verify=False for SVU website certificate issues if needed, strictly speaking we should try verify=True first or warn
        print(f"Fetching {url}...")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, verify=False, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove scripts and styles
        for script in soup(["script", "style", "nav", "footer", "iframe"]):
            script.decompose()
            
        # Get text
        text = soup.get_text(separator=' ', strip=True)
        # Clean up whitespace
        text = ' '.join(text.split())
        return text[:15000] # Limit context window, increased slightly
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return None

def generate_faqs(text, url):
    print("Generating FAQs using AI...")
    model = genai.GenerativeModel(MODEL_NAME)
    prompt = f"""
    Analyze the following text scraped from {url}.
    Extract key facts, dates, names, roles, and procedures.
    Generate 5 to 10 high-quality Question-Answer pairs (FAQs) based ONLY on this text.
    Format the output as a JSON array of objects, where each object has "question" and "answer" keys.
    Do not add any markdown formatting like ```json. just return the raw JSON array.
    
    Text:
    {text}
    """
    
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
            
        return json.loads(content)
    except Exception as e:
        print(f"Error generating FAQS: {e}")
        return []

def update_database(new_faqs, url):
    print(f"Updating {DATA_FILE}...")
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
        
    # Append with a unique ID
    existing_ids = set(item.get("id") for item in data["faqs"] if item.get("id"))
    
    added_count = 0
    for faq in new_faqs:
        # Create a simple deterministic-ish ID or just random
        base_id = f"auto-{int(os.urandom(4).hex(), 16)}"
        while base_id in existing_ids:
             base_id = f"auto-{int(os.urandom(4).hex(), 16)}"
             
        faq["id"] = base_id
        faq["source"] = url
        data["faqs"].append(faq)
        added_count += 1
        
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Successfully added {added_count} FAQs to {DATA_FILE}")

def main():
    parser = argparse.ArgumentParser(description="Scrape a URL and add FAQs to knowledge base.")
    parser.add_argument("url", help="The URL to scrape")
    args = parser.parse_args()
    
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    text = get_page_text(args.url)
    if text:
        print(f"Extracted {len(text)} characters.")
        faqs = generate_faqs(text, args.url)
        if faqs:
            print(f"Generated {len(faqs)} FAQs.")
            update_database(faqs, args.url)
        else:
            print("No FAQs could be generated from the content.")
    else:
        print("Failed to retrieve content.")

if __name__ == "__main__":
    main()
