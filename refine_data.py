import json
import os

DATA_FILE = "data.json"

def classify_category(question, answer):
    text = (question + " " + answer).lower()
    
    if any(k in text for k in ["vice-chancellor", "rector", "registrar", "dean", "principal", "executive council", "senate", "officer", "administration", "leadership"]):
        return "Administration"
    if any(k in text for k in ["admission", "apply", "fee", "scholarship", "eligibility", "entrance", "exam", "rank", "seat", "application"]):
        return "Admissions"
    if any(k in text for k in ["course", "program", "syllabus", "curriculum", "degree", "b.tech", "m.tech", "phd", "academic", "department", "faculty", "class"]):
        return "Academics"
    if any(k in text for k in ["hostel", "library", "wifi", "transport", "bus", "sports", "gym", "canteen", "lab", "facility", "infrastructure", "building", "campus"]):
        return "Facilities"
    if any(k in text for k in ["address", "phone", "email", "location", "where is", "contact", "reach"]):
        return "Contact & Location"
    if any(k in text for k in ["placement", "job", "recruit", "salary", "package", "internship", "career"]):
        return "Placements"
    if any(k in text for k in ["history", "established", "motto", "vision", "mission", "naac", "ranking", "about", "founder"]):
        return "General Info"
        
    return "Other"

def refine_data():
    if not os.path.exists(DATA_FILE):
        print(f"File {DATA_FILE} not found.")
        return

    with open(DATA_FILE, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print("Invalid JSON.")
            return

    if "faqs" not in data:
        print("No FAQs found.")
        return

    print(f"Processing {len(data['faqs'])} FAQs...")
    
    refined_faqs = []
    seen_questions = set()

    for item in data["faqs"]:
        question = item.get("question", "").strip()
        answer = item.get("answer", "").strip()
        
        if not question or not answer:
            continue
            
        # Deduplication (simple exact match on question)
        # We also check if a very similar question exists? For now, exact match specific to casing.
        q_lower = question.lower()
        if q_lower in seen_questions:
            continue
        seen_questions.add(q_lower)
        
        # Categorize
        category = classify_category(question, answer)
        
        # Structure
        new_item = {
            "id": item.get("id", f"faq-{len(refined_faqs)}"),
            "category": category,
            "question": question,
            "answer": answer,
            "source": item.get("source", "Manual/Legacy")
        }
        refined_faqs.append(new_item)

    # Sort by category for better readability in the file
    refined_faqs.sort(key=lambda x: x["category"])

    data["faqs"] = refined_faqs
    
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)
        
    print(f"Refinement complete. Saved {len(refined_faqs)} clean, categorized FAQs to {DATA_FILE}.")

if __name__ == "__main__":
    refine_data()
