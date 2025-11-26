import re
import csv
import pdfplumber

PDF_PATH = "recommendation_generation/standard-treatment-guidelines.pdf"
OUTPUT_CSV = "standard-treatment-guidelines.csv"

# Regex patterns
DISEASE_HEADING = re.compile(r'^[A-Z][A-Z \-/()0-9]{4,}$')  # FULL CAPS lines
MANAGEMENT_START = re.compile(r'\b(MANAGEMENT|TREATMENT|MANAGEMENT AND TREATMENT)\b', re.IGNORECASE)
MANAGEMENT_END = re.compile(r'\b(FOLLOW[- ]?UP|REFERRAL|REFER)\b', re.IGNORECASE)


def extract_pdf_text(pdf_path):
    print("> Extracting raw text from PDF ...")
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n".join(pages)


def split_into_disease_sections(text):
    print("> Splitting text into disease sections ...")
    lines = text.split("\n")

    sections = []
    current_disease = None
    current_text = []

    for line in lines:
        if DISEASE_HEADING.match(line.strip()):
            # Save previous section
            if current_disease and current_text:
                sections.append((current_disease, "\n".join(current_text).strip()))

            # Start new section
            current_disease = line.strip()
            current_text = []
        else:
            if current_disease:
                current_text.append(line)

    # Save the final section
    if current_disease and current_text:
        sections.append((current_disease, "\n".join(current_text).strip()))

    return sections


def extract_treatment_text(section_text):
    # Find start
    start_match = MANAGEMENT_START.search(section_text)
    if not start_match:
        return None

    start_idx = start_match.start()

    # Find end
    end_match = MANAGEMENT_END.search(section_text[start_idx:])
    if end_match:
        end_idx = start_idx + end_match.start()
        return section_text[start_idx:end_idx].strip()

    return section_text[start_idx:].strip()


def build_csv(sections):
    print("> Extracting treatment parts and writing CSV ...")

    rows = []
    for disease, text in sections:
        treatment = extract_treatment_text(text)
        if treatment:
            rows.append([disease, treatment])

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["disease", "treatment"])
        writer.writerows(rows)

    print(f"> Done! Extracted {len(rows)} diseases into {OUTPUT_CSV}")


if __name__ == "__main__":
    raw_text = extract_pdf_text(PDF_PATH)
    sections = split_into_disease_sections(raw_text)
    build_csv(sections)
