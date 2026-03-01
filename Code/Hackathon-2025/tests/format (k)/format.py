import re
import spacy

try:
    nlp = spacy.load("en_core_sci_sm")  # get scispacy
except:
    nlp = spacy.load("en_core_web_sm")

raw_patterns = {
    "Patient Name": r"Name[:\-]?\s*(.*)",
    "Age": r"Age[:\-]?\s*(\d+)",
    "Gender": r"(?:Gender|Sex)[:\-]?\s*(Male|Female|Other)",
    "Medications": r"Medications[:\-]?\s*(.*)",
    "Allergies": r"Allergies[:\-]?\s*(.*)",
}

compiled_patterns = {k: re.compile(v, re.IGNORECASE) for k, v in raw_patterns.items()}


def parse_clinical_text(text):
    structured_data = {
        k: None
        for k in [
            "Patient Name",
            "Age",
            "Gender",
            "Chief Complaint",
            "History of Present Illness",
            "Past Medical History",
            "Medications",
            "Allergies",
            "Assessment",
            "Plan",
        ]
    }

    for key, pattern in compiled_patterns.items():
        match = pattern.search(text)
        if match:
            structured_data[key] = match.group(1).strip()

    if not structured_data["Age"]:
        age_match = re.search(r"\b(\d{1,2})\s*year[s]?\s*old\b", text, re.IGNORECASE)
        if age_match:
            structured_data["Age"] = age_match.group(1)

    if not structured_data["Chief Complaint"]:
        cc_match = re.search(r"presents with (.*?)(?:\.|\n)", text, re.IGNORECASE)
        if cc_match:
            structured_data["Chief Complaint"] = cc_match.group(1)

    if not structured_data["Past Medical History"]:
        pmh_match = re.search(r"history of (.*?)(?:\.|\n)", text, re.IGNORECASE)
        if pmh_match:
            structured_data["Past Medical History"] = pmh_match.group(1)

    doc = nlp(text)
    meds = [ent.text for ent in doc.ents if ent.label_ in ["DRUG", "CHEMICAL"]]
    diagnoses = [ent.text for ent in doc.ents if ent.label_ in ["DISEASE", "CONDITION"]]

    if meds and not structured_data["Medications"]:
        structured_data["Medications"] = ", ".join(set(meds))

    if diagnoses and not structured_data["Assessment"]:
        structured_data["Assessment"] = ", ".join(set(diagnoses))

    return structured_data


def create_structured_note(structured_data):
    output = ["--- Structured Clinical Note ---"]
    for key, value in structured_data.items():
        output.append(f"{key}: {value if value else 'Not specified'}")
    return "\n".join(output)


if __name__ == "__main__":
    with open("raw_clinical_note.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    structured = parse_clinical_text(raw_text)
    formatted_note = create_structured_note(structured)

    with open("structured_clinical_note.txt", "w", encoding="utf-8") as f:
        f.write(formatted_note)

    print("Structured note created successfully!")
