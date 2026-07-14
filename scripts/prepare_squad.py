import os
import json
import random
import requests

# SQuAD v2.0 Official Validation Dataset URL
SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
OUTPUT_DIR = "data/benchmark"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "golden_dataset.json")

def download_squad():
    """Downloads the SQuAD v2.0 dataset directly from the official source."""
    print(f"[*] Downloading SQuAD v2.0 from {SQUAD_URL}...")
    response = requests.get(SQUAD_URL)
    response.raise_for_status()
    return response.json()

def prepare_golden_dataset(squad_data):
    """
    Extracts exactly 20 diverse questions:
    - 15 Answerable questions (to test retrieval and extraction)
    - 5 Unanswerable questions (to test hallucination resistance)
    """
    answerable = []
    unanswerable = []
    
    print("[*] Parsing documents and extracting the Golden Dataset...")
    
    for article in squad_data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                query = qa['question']
                is_impossible = qa.get('is_impossible', False)
                
                # 15 Cevaplanabilir Soru
                if not is_impossible and len(answerable) < 15:
                    answer = qa['answers'][0]['text']
                    answerable.append({
                        "context": context, 
                        "query": query, 
                        "answer": answer,
                        "type": "answerable"
                    })
                # 5 Cevaplanamaz Soru (Halüsinasyon Testi)
                elif is_impossible and len(unanswerable) < 5:
                    unanswerable.append({
                        "context": context, 
                        "query": query, 
                        "answer": "I don't know.", # Expected strict behavior
                        "type": "unanswerable"
                    })
                    
                if len(answerable) == 15 and len(unanswerable) == 5:
                    break
            if len(answerable) == 15 and len(unanswerable) == 5:
                break
        if len(answerable) == 15 and len(unanswerable) == 5:
            break
            
    # Soruları karıştır (bias oluşmaması için)
    golden_dataset = answerable + unanswerable
    random.shuffle(golden_dataset)
    return golden_dataset

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        raw_data = download_squad()
        golden_data = prepare_golden_dataset(raw_data)
        
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(golden_data, f, indent=4, ensure_ascii=False)
            
        print(f"[SUCCESS] Saved {len(golden_data)} golden questions to '{OUTPUT_FILE}'")
        print("[INFO] Dataset distribution: 15 Answerable, 5 Unanswerable (Hallucination traps).")
    
    except Exception as e:
        print(f"[ERROR] Failed to prepare dataset: {e}")