import csv
import string
import re

from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDFS, SKOS
import Levenshtein


SNOMED_TTL_FILE = "snomed-ct-20221231-mini.ttl"  
INPUT_CSV_FILE = "input_descriptions.csv"        
OUTPUT_CSV_FILE = "output_matches.csv"           

# Load the SNOMED Turtle file and extract all the concept IDs and their labels
def parse_snomed_turtle(ttl_path):
    g = Graph()
    print(f"Parsing Turtle file: {ttl_path} ...")
    g.parse(ttl_path, format="turtle")
    print(f"Graph has {len(g)} total RDF triples.\n")

    concept_to_labels = {}
    label_predicates = [RDFS.label, SKOS.altLabel, SKOS.prefLabel]

    # Loop through everything in the graph and grab labels
    for s, p, o in g:
        if p in label_predicates and isinstance(s, URIRef) and isinstance(o, Literal):
            concept_id = extract_snomed_id(s)
            if concept_id is not None:
                text_label = str(o)
                concept_to_labels.setdefault(concept_id, []).append(text_label)

    print(f"Extracted {len(concept_to_labels)} SNOMED concepts with ≥1 label.\n")
    return concept_to_labels

# Just get the SNOMED ID (the number at the end of the URI)
def extract_snomed_id(uri):
    if not isinstance(uri, URIRef):
        return None
    match = re.search(r"/id/(\d+)$", str(uri))
    if match:
        return match.group(1)
    else:
        return None


# Create an index for blocking by breaking descriptions into tokens
def build_token_index(concept_dict):
    token_index = {}

    for concept_id, labels in concept_dict.items():
        # Combine all the labels into one big text blob
        combined_text = " ".join(labels)
        # Tokenize the text and add to the index
        tokens = tokenize_for_blocking(combined_text)
        for t in tokens:
            token_index.setdefault(t, set()).add(concept_id)

    return token_index


# Clean up the token index to remove overly large blocks
def block_cleaning(token_index, max_size=2000):
    if max_size is None:
        return token_index  # Skip cleaning if no size limit is set
    cleaned = {}
    for token, concept_ids in token_index.items():
        if len(concept_ids) <= max_size:  # Keep only "reasonable-sized" blocks
            cleaned[token] = concept_ids
    return cleaned


# Turn a string into a set of lowercase tokens (no punctuation, etc.)
def tokenize_for_blocking(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    return set(tokens)

# Preprocess text for similarity matching (normalize casing and punctuation)
def normalize_text_for_levenshtein(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

# Calculate how similar two strings are (higher score = more similar)
def levenshtein_similarity(str1, str2):
    return Levenshtein.ratio(str1, str2)


# Match each local description to a SNOMED code
def find_best_snomed_match(description, token_index, concept_dict, sim_threshold=0.7):
    # Step 1: Tokenize the description and find matching tokens in the index
    desc_tokens = tokenize_for_blocking(description)
    candidate_codes = set()
    for t in desc_tokens:
        if t in token_index:
            candidate_codes.update(token_index[t])

    # If no candidates, just return "no match"
    if not candidate_codes:
        return None, 0.0

    best_code = None
    best_sim = 0.0
    desc_for_lev = normalize_text_for_levenshtein(description)

    # Step 2: Compare the description to all the labels of the candidate codes
    for code in candidate_codes:
        label_list = concept_dict[code]
        max_label_sim = 0.0
        for lbl in label_list:
            lbl_for_lev = normalize_text_for_levenshtein(lbl)
            sim = levenshtein_similarity(desc_for_lev, lbl_for_lev)
            if sim > max_label_sim:
                max_label_sim = sim

        # Keep track of the best match
        if max_label_sim > best_sim:
            best_sim = max_label_sim
            best_code = code

    # Only return matches that are above the similarity threshold
    if best_sim >= sim_threshold:
        return best_code, best_sim
    else:
        return None, best_sim


def main():
    # Step 1: Load and process the SNOMED data
    concept_dict = parse_snomed_turtle(SNOMED_TTL_FILE)
    raw_index = build_token_index(concept_dict)
    cleaned_index = block_cleaning(raw_index, max_size=2000)

    print("=== Matching Local Descriptions to SNOMED (Levenshtein) ===")

    # Step 2: Open the input file and write matches to the output file
    with open(INPUT_CSV_FILE, "r", encoding="utf-8") as f_in, open(OUTPUT_CSV_FILE, "w", encoding="utf-8", newline="") as f_out:
        reader = csv.DictReader(f_in)
        fieldnames = ["id", "description", "matched_snomed", "similarity_score"]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            record_id = row["id"]
            description = row["description"].strip()

            code, sim = find_best_snomed_match(
                description=description,
                token_index=cleaned_index,
                concept_dict=concept_dict,
                sim_threshold=0.7  # Match threshold (higher = stricter)
            )

            # Write the match (or "No match") to the CSV
            if code and sim >= 0.7:
                print(f"[{record_id}] '{description}' -> SNOMED {code}, sim={sim:.2f}")
                writer.writerow({
                    "id": record_id,
                    "description": description,
                    "matched_snomed": code,
                    "similarity_score": f"{sim:.2f}"
                })
            else:
                print(f"[{record_id}] '{description}' -> No match (best sim={sim:.2f})")
                writer.writerow({
                    "id": record_id,
                    "description": description,
                    "matched_snomed": "No match",
                    "similarity_score": "0.00"
                })

    print(f"\nMatching results saved to '{OUTPUT_CSV_FILE}'.\nDone.")

if __name__ == "__main__":
    main()