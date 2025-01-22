import csv
import string
import re
from collections import defaultdict

from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDFS, SKOS
import Levenshtein


SNOMED_TTL_FILE = "snomed-ct-20221231-mini.ttl"
INPUT_CSV_FILE = "input_descriptions.csv"
OUTPUT_MATCH_CSV_FILE = "output_matches.csv"
OUTPUT_CLUSTER_CSV_FILE = "output_clusters.csv"

# Parsing and Indexing SNOMED

def parse_snomed_turtle(ttl_path):
    """
    Load the SNOMED Turtle file and extract concept IDs + labels.
    """
    g = Graph()
    print(f"Parsing Turtle file: {ttl_path} ...")
    g.parse(ttl_path, format="turtle")
    print(f"Graph has {len(g)} total RDF triples.\n")

    concept_to_labels = {}
    label_preds = [RDFS.label, SKOS.altLabel, SKOS.prefLabel]

    # Look for each triple with a label predicate
    for s, p, o in g:
        if p in label_preds and isinstance(s, URIRef) and isinstance(o, Literal):
            concept_id = extract_snomed_id(s)
            if concept_id:
                text_label = str(o)
                concept_to_labels.setdefault(concept_id, []).append(text_label)

    print(f"Extracted {len(concept_to_labels)} SNOMED concepts with ≥1 label.\n")
    return concept_to_labels


def extract_snomed_id(uri):
    """
    Grab the numeric SNOMED ID from the URI (e.g. '/id/12345').
    """
    if not isinstance(uri, URIRef):
        return None
    match = re.search(r"/id/(\d+)$", str(uri))
    if match:
        return match.group(1)
    return None

# Blocking (tokenizing) and Block Cleaning
def build_token_index(concept_dict):
    """
    For each SNOMED concept, combine all labels into a big text chunk.
    Then tokenize it so we can do a 'blocking' style approach.
    We'll store: token -> set of concept_ids
    """
    token_index = {}

    for concept_id, labels in concept_dict.items():
        combined_text = " ".join(labels)
        tokens = tokenize_for_blocking(combined_text)
        for t in tokens:
            token_index.setdefault(t, set()).add(concept_id)

    return token_index

def tokenize_for_blocking(text):
    """
    Lowercase, strip punctuation, then split into tokens.
    Returns a set of tokens (unique terms).
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return set(text.split())

def block_cleaning(token_index, max_size=2000):
    """
    If a token is too generic (it appears in a huge number of concepts),
    it will be removed from the index to avoid generating massive blocks.
    """
    if max_size is None:
        return token_index  # skipping if not needed

    cleaned = {}
    for token, concept_ids in token_index.items():
        if len(concept_ids) <= max_size:
            cleaned[token] = concept_ids
    return cleaned


# Similarity Computation
def normalize_text_for_levenshtein(text):
    """
    Normalize text before computing Levenshtein similarity:
    lowercase, remove punctuation, strip whitespace.
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

def levenshtein_similarity(str1, str2):
    """
    Higher score = more similar (ratio in [0..1]).
    """
    return Levenshtein.ratio(str1, str2)

# Matching

def find_best_snomed_match(description, token_index, concept_dict, sim_threshold=0.7):
    """
    1) Tokenize the local description to find candidate SNOMED codes via the index.
    2) Among those candidates, compute the Levenshtein similarity to each code's labels.
    3) Return the code (and similarity) that best matches, if above the threshold.
    """
    # Step 1: gather candidate SNOMED codes based on shared tokens
    desc_tokens = tokenize_for_blocking(description)
    candidate_codes = set()
    for t in desc_tokens:
        if t in token_index:
            candidate_codes.update(token_index[t])

    if not candidate_codes:
        return None, 0.0  # no candidates at all

    # Step 2: compare this description to each candidate's labels
    best_code = None
    best_sim = 0.0
    desc_normal = normalize_text_for_levenshtein(description)

    for code in candidate_codes:
        labels = concept_dict[code]
        max_label_sim = 0.0
        for lbl in labels:
            lbl_normal = normalize_text_for_levenshtein(lbl)
            sim = levenshtein_similarity(desc_normal, lbl_normal)
            if sim > max_label_sim:
                max_label_sim = sim

        # track the top overall match
        if max_label_sim > best_sim:
            best_sim = max_label_sim
            best_code = code

    # Step 3: return the match if above threshold
    if best_sim >= sim_threshold:
        return best_code, best_sim
    else:
        return None, best_sim

# Simple Clustering
def cluster_by_snomed(match_results):
    """
    Groups local descriptions together if they matched the same SNOMED code.
    This is a very basic clustering approach:
      cluster_id = SNOMED code
      items = list of (record_id, local_description, similarity)
    """
    clusters = defaultdict(list)
    for result in match_results:
        record_id, local_desc, code, sim = result
        if code != "No match":
            clusters[code].append((record_id, local_desc, sim))
    return clusters


def save_clusters_to_csv(clusters, output_csv):
    """
    Writes out a CSV where each row is:
        SNOMED_CODE, CLUSTER_SIZE, RECORD_IDS, DESCRIPTIONS, SIMILARITIES
    to give a quick overview of how local descriptions are grouped.
    """
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["snomed_code", "cluster_size", "record_ids", "descriptions", "similarities"])

        for code, items in clusters.items():
            # Each item is (record_id, local_desc, sim)
            cluster_size = len(items)
            record_ids = [str(i[0]) for i in items]
            descriptions = [i[1] for i in items]
            sims = [f"{i[2]:.2f}" for i in items]

            writer.writerow([
                code,
                cluster_size,
                ";".join(record_ids),
                ";".join(descriptions),
                ";".join(sims)
            ])

def main():
     # Load SNOMED data, create token index, do block cleaning
    concept_dict = parse_snomed_turtle(SNOMED_TTL_FILE)
    raw_index = build_token_index(concept_dict)
    cleaned_index = block_cleaning(raw_index, max_size=2000)

    print("=== Matching Local Descriptions to SNOMED (with optional clustering) ===\n")

    match_results = []  # store everything for later clustering

    #  Read local descriptions, find best SNOMED code, write to CSV
    with open(INPUT_CSV_FILE, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_MATCH_CSV_FILE, "w", encoding="utf-8", newline="") as f_out:

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
                sim_threshold=0.7  # tweak as desired
            )

            if code and sim >= 0.7:
                print(f"[{record_id}] '{description}' -> SNOMED {code}, sim={sim:.2f}")
                writer.writerow({
                    "id": record_id,
                    "description": description,
                    "matched_snomed": code,
                    "similarity_score": f"{sim:.2f}"
                })
                match_results.append((record_id, description, code, sim))
            else:
                print(f"[{record_id}] '{description}' -> No match (best sim={sim:.2f})")
                writer.writerow({
                    "id": record_id,
                    "description": description,
                    "matched_snomed": "No match",
                    "similarity_score": f"{sim:.2f}"
                })
                match_results.append((record_id, description, "No match", sim))

    print(f"\nMatches saved to '{OUTPUT_MATCH_CSV_FILE}'.\n")

    # 3) Perform a simple clustering step: group by matched SNOMED code
    clusters = cluster_by_snomed(match_results)

    # 4) Save cluster info to a separate CSV
    save_clusters_to_csv(clusters, OUTPUT_CLUSTER_CSV_FILE)
    print(f"Clusters saved to '{OUTPUT_CLUSTER_CSV_FILE}'.\nDone.")


if __name__ == "__main__":
    main()
