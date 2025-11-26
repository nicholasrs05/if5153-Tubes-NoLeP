import pdfplumber
import re
import pandas as pd
from typing import Dict, List, Tuple


PDF_PATH = "../data/AZ_Common_symptom_answer_guide.pdf"   # sesuaikan lokasi file PDF-mu
OUTPUT_CSV_PATH = "cause_from_pdf.csv"


def extract_full_text_from_pdf(pdf_path: str) -> str:
    
    all_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            all_text.append(page_text)
    return "\n".join(all_text)


def build_heading_index(
    full_text: str,
    heading_variants: Dict[str, List[str]],
) -> List[Tuple[int, str]]:
    
    text_upper = full_text.upper()
    hits = []

    # heading_variants: key = heading_label, value = list of keyword patterns
    for heading_key, variants in heading_variants.items():
        for pattern in variants:
            pattern_upper = pattern.upper()
            idx = text_upper.find(pattern_upper)
            if idx != -1:
                hits.append((idx, heading_key))
                break  # cukup ambil match pertama untuk heading ini

    # sort berdasarkan posisi kemunculan
    hits.sort(key=lambda x: x[0])
    return hits


def slice_sections_by_headings(
    full_text: str,
    heading_positions: List[Tuple[int, str]],
) -> Dict[str, str]:
    """
    Potong full_text menjadi beberapa section berdasarkan heading_positions.
    heading_positions: list of (start_idx, heading_key) sorted.
    Return: dict {heading_key: section_text}
    """
    sections = {}
    n = len(heading_positions)

    for i, (start_idx, heading_key) in enumerate(heading_positions):
        if i < n - 1:
            end_idx = heading_positions[i + 1][0]
        else:
            end_idx = len(full_text)

        section_text = full_text[start_idx:end_idx].strip()
        sections[heading_key] = section_text

    return sections


def main():
    print(f"[INFO] Extracting text from PDF: {PDF_PATH}")
    full_text = extract_full_text_from_pdf(PDF_PATH)

    
    CATEGORY_TO_HEADINGS = {
        "Back pain": ["BACK PAIN"],
        "Cough": ["COUGH"],
        "Feeling dizzy": ["DIZZINESS"],
        "Foot ache": ["FOOT OR ANKLE PAIN", "FOOT PAIN"],
        "Joint pain": ["JOINT PAIN"],
        "Knee pain": ["KNEE PAIN"],
        "Neck pain": ["NECK PROBLEMS", "NECK PAIN"],
        "Stomach ache": ["ABDOMINAL PAIN", "ABDOMINAL PAIN (ADULT)"],
        "Hard to breath": ["BREATHING PROBLEMS", "SHORTNESS OF BREATH"],
        "Emotional pain": ["DEPRESSION, SUICIDAL THOUGHTS, OR ANXIETY", "DEPRESSION"],
        "Skin issue": ["SKIN PROBLEMS", "RASH"],
        "Open wound": ["CUTS AND SCRAPES", "LACERATIONS"],
        "Infected wound": ["SKIN INFECTIONS", "CELLULITIS"],
        "Head ache": ["HEADACHE"],
        "Body feels weak": ["WEAKNESS", "FATIGUE"],
        "Muscle pain": ["MUSCLE PAIN", "MUSCLE ACHES"],
        "Shoulder pain": ["SHOULDER PAIN"],
        "Heart hurts": ["CHEST PAIN", "CHEST DISCOMFORT"],
    }

    
    heading_variants = {}
    for heading_label, variants in CATEGORY_TO_HEADINGS.items():
        
        heading_variants[heading_label] = variants

    print("[INFO] Building heading index from full text...")
    heading_positions = build_heading_index(full_text, heading_variants)

    if not heading_positions:
        print("[WARN] No headings found. You probably need to adjust CATEGORY_TO_HEADINGS to match the actual PDF headings.")
        return

    print("[INFO] Found the following heading positions:")
    for pos, key in heading_positions:
        print(f" - {key} at char index {pos}")

    print("[INFO] Slicing full text into sections...")
    sections_by_heading = slice_sections_by_headings(full_text, heading_positions)

   
    rows = []
    for category, variants in CATEGORY_TO_HEADINGS.items():
        
        heading_key = category  
        section_text = sections_by_heading.get(heading_key, "")

        if not section_text:
            print(f"[WARN] No section text found for category '{category}'. Check heading mapping.")
            continue

        rows.append({
            "Category": category,
            "CauseText": section_text,
        })

    if not rows:
        print("[ERROR] No rows to save. Aborting.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"[INFO] Saved CSV to: {OUTPUT_CSV_PATH}")
    print(df.head())


if __name__ == "__main__":
    main()
