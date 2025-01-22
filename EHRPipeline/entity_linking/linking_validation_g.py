import csv
import re
import string
import xml.etree.ElementTree as ET
import Levenshtein

from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDFS, SKOS

# Files and thresholds
INPUT_XML = "SNOMEDANDICD_results.xml"
ICD_CSV = "D_ICD_DIAGNOSES.csv"          # Contains ICD codes and their descriptions
SNOMED_TTL = "snomed-ct-20221231-mini.ttl"
OUTPUT_CSV = "icd_snomed_validation.csv"

# We keep the same general threshold for the composite similarity
SIM_THRESHOLD = 0.5

def normalize(text):
    """
    We'll keep this straightforward: turn everything lowercase,
    remove punctuation, and trim spaces. That should help standardize text.
    """
    if not text:
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()

def levenshtein_similarity(a, b):
    """
    This is your existing string similarity method (ratio).
    Basically returns a value from 0.0 to 1.0.
    """
    return Levenshtein.ratio(a, b)

def tokenize(text):
    """
    Let's split on whitespace to get tokens, then
    remove any leftover empty strings. We could add
    additional logic if needed, but let's keep it simple.
    """
    if not text:
        return []
    return [t for t in text.split() if t]

def weighted_jaccard_similarity(text_a, text_b):
    """
    This function tries to capture the notion of Weighted Jaccard that appears
    often in entity resolution works. The idea is to see how token frequencies
    overlap, instead of just checking set membership.

    We'll build frequency dicts for each text, then sum(min(freqA, freqB)) / sum(max(freqA, freqB)).

    If either text is empty, we just return 0.0
    """
    if not text_a or not text_b:
        return 0.0

    # Tokenize
    tokens_a = tokenize(text_a)
    tokens_b = tokenize(text_b)

    # Count frequencies
    freq_a = {}
    freq_b = {}

    for tok in tokens_a:
        freq_a[tok] = freq_a.get(tok, 0) + 1
    for tok in tokens_b:
        freq_b[tok] = freq_b.get(tok, 0) + 1

    # Gather union of all tokens
    all_tokens = set(freq_a.keys()).union(set(freq_b.keys()))
    if not all_tokens:
        return 0.0

    sum_min = 0
    sum_max = 0
    for tok in all_tokens:
        sum_min += min(freq_a.get(tok, 0), freq_b.get(tok, 0))
        sum_max += max(freq_a.get(tok, 0), freq_b.get(tok, 0))

    if sum_max == 0:
        return 0.0
    return float(sum_min) / float(sum_max)

def composite_similarity(icd_text, snomed_text, alpha=0.5):
    """
    Combine Weighted Jaccard and Levenshtein into a single metric.
    alpha controls how much weight is given to Weighted Jaccard.
    (1 - alpha) is the weight of Levenshtein similarity.

    If alpha=0.5, it balances both similarity metrics equally.
    """
    # normalize the inputs
    norm_icd = normalize(icd_text)
    norm_snomed = normalize(snomed_text)

    lev_sim = levenshtein_similarity(norm_icd, norm_snomed)
    jaccard_sim = weighted_jaccard_similarity(norm_icd, norm_snomed)

    # Weighted linear combination
    return alpha * jaccard_sim + (1 - alpha) * lev_sim

def extract_snomed_id(uri):
    """
    Given a SNOMED URI like 'http://snomed.info/id/212505002',
    we want to snag the trailing numeric portion.
    """
    match = re.search(r"/id/(\d+)$", uri)
    if match:
        return match.group(1)
    return None

def extract_icd9_code(raw):
    """
    Grab the real ICD code from something like 'icd9#412' or 'icd10#AB123'.
    We'll keep the pattern simple with 'icd9#' or 'icd10#'.
    """
    pattern = r"(?:icd9#|icd10#)(.*)"
    match = re.match(pattern, raw, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return None

def load_icd_descriptions(csv_path):
    """
    Read the CSV that has ICD codes + short & long descriptions.
    We return a dict with the code as key, and short/long as separate strings.
    """
    icd_dict = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row["icd9_code"].strip()
            short_desc = row["short_title"].strip()
            long_desc = row["long_title"].strip()
            icd_dict[code] = {
                "short": short_desc,
                "long": long_desc
            }
    return icd_dict

def parse_snomed_turtle(g):
    """
    Loads the SNOMED TTL file into an RDF Graph. Then we'll grab labels from
    RDFS.label, SKOS.altLabel, or SKOS.prefLabel. We'll store them in a dict:
       concept_to_labels[snomed_id] = [ label1, label2, ... ]
    """
    #print("Loading SNOMED TTL:", ttl_path)
    #g = Graph()
    #g.parse(ttl_path, format="turtle")
    print(f"Cool, the graph has {len(g)} RDF triples.\n")

    concept_to_labels = {}
    label_predicates = [RDFS.label, SKOS.altLabel, SKOS.prefLabel]

    for s, p, o in g:
        if p in label_predicates and isinstance(s, URIRef) and isinstance(o, Literal):
            snomed_id = extract_snomed_id(str(s))
            if snomed_id:
                concept_to_labels.setdefault(snomed_id, []).append(str(o))

    print(f"Extracted label info for {len(concept_to_labels)} SNOMED concepts.\n")
    return concept_to_labels

def parse_icd_snomed_links(xml_file):
    """
    We look for <Description> elements that have <hasCode> and <Linked rdf:resource=...>.
    We'll make a list like:
        {
            'icd_raw': 'icd9#412',
            'snomed_uri': 'http://snomed.info/id/212505002'
        }
    so we can process them later.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    links = []
    for desc_el in root.findall(".//{*}Description"):
        has_code_el = desc_el.find(".//{*}hasCode")
        linked_el = desc_el.find(".//{*}Linked")

        if has_code_el is not None and linked_el is not None:
            icd_raw_text = has_code_el.text
            snomed_res = linked_el.attrib.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource")

            if icd_raw_text and snomed_res:
                links.append({
                    "icd_raw": icd_raw_text.strip(),
                    "snomed_uri": snomed_res.strip()
                })
    return links

def LinkingValidator(KG):
    # 1) Load ICD descriptions
    icd_descriptions = load_icd_descriptions(ICD_CSV)

    # 2) Parse SNOMED TTL for concept->label(s)
    snomed_labels = parse_snomed_turtle(SNOMED_TTL)

    # 3) Parse the XML with ICD->SNOMED pairs
    links = parse_icd_snomed_links(KG)
    print(f"Found {len(links)} ICD->SNOMED mappings in the XML.\n")

    # 4) We'll combine Weighted Jaccard and Levenshtein for a more thorough similarity.
    #    We'll store the best label match for each SNOMED concept.
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow([
            "icd_raw",
            "icd_extracted",
            "icd_text",
            "snomed_uri",
            "snomed_extracted",
            "best_snomed_label",
            "composite_similarity",
            "verdict"
        ])

        for link in links:
            icd_raw_value = link["icd_raw"]
            snomed_uri_value = link["snomed_uri"]

            # Extract codes/IDs
            icd_code = extract_icd9_code(icd_raw_value)
            snomed_id = extract_snomed_id(snomed_uri_value)

            # Build a text from short+long desc
            icd_text = ""
            if icd_code and icd_code in icd_descriptions:
                short_t = icd_descriptions[icd_code]["short"]
                long_t = icd_descriptions[icd_code]["long"]
                icd_text = f"{short_t} {long_t}".strip()

            # Candidate labels for SNOMED
            candidate_labels = snomed_labels.get(snomed_id, []) if snomed_id else []

            # Find the label with highest composite similarity
            best_sim = 0.0
            best_label = ""
            for label in candidate_labels:
                sim = composite_similarity(icd_text, label, alpha=0.5)
                if sim > best_sim:
                    best_sim = sim
                    best_label = label

            verdict = "OK" if best_sim >= SIM_THRESHOLD else "Suspicious"

            writer.writerow([
                icd_raw_value,
                icd_code if icd_code else "",
                icd_text,
                snomed_uri_value,
                snomed_id if snomed_id else "",
                best_label,
                f"{best_sim:.3f}",
                verdict
            ])

    print(f"Wrote the matching results to '{OUTPUT_CSV}'. All done!")

#if __name__ == "__main__":
 #   main()