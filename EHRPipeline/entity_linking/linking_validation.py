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
SIM_THRESHOLD = 0.7


def normalize(text):
    """
    Takes a string, lowercases it, removes punctuation, and trims whitespace.
    Used to standardize text before computing similarity.
    """
    if not text:
        return ""
    # Make everything lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Trim extra whitespace
    return text.strip()


def levenshtein_similarity(a, b):
    """
    Wraps Levenshtein.ratio to compute the similarity ratio
    between two strings a and b. Values range from 0 to 1.
    """
    return Levenshtein.ratio(a, b)


def extract_snomed_id(uri):
    """
    Given a SNOMED URI (e.g., 'http://snomed.info/id/212505002'),
    extract the final numeric part (e.g., '212505002').
    """
    match = re.search(r"/id/(\d+)$", uri)
    if match:
        return match.group(1)
    return None


def extract_icd9_code(raw):
    """
    Parses out the actual ICD code from a string like 'icd9#412' or 'icd10#AB123'.
    For example:
        'icd9#412' -> '412'
        'icd10#AB123' -> 'AB123'
    """
    pattern = r"(?:icd9#|icd10#)(.*)"
    match = re.match(pattern, raw, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def load_icd_descriptions(csv_path):
    """
    Reads the CSV file containing ICD codes and their short/long descriptions.
    
    Returns a dictionary:
    {
        'ICD_CODE': {
            'short': <short description>,
            'long':  <long description>
        },
        ...
    }
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
    Loads the SNOMED TTL file into an RDF graph, then builds and returns
    a dictionary of:
    
        concept_to_labels[snomed_id] = [list of labels]
    """
    # print("Parsing SNOMED TTL:", ttl_path)
    # g = Graph()
    # g.parse(ttl_path, format="turtle")
    print(f"Graph has {len(g)} RDF triples.\n")

    concept_to_labels = {}
    label_predicates = [RDFS.label, SKOS.altLabel, SKOS.prefLabel]

    for subject, predicate, obj in g:
        if predicate in label_predicates and isinstance(subject, URIRef) and isinstance(obj, Literal):
            snomed_id = extract_snomed_id(str(subject))
            if snomed_id:
                concept_to_labels.setdefault(snomed_id, []).append(str(obj))

    print(f"Extracted {len(concept_to_labels)} SNOMED concepts.\n")
    return concept_to_labels


def parse_icd_snomed_links(xml_file):
    """
    Reads the XML file containing ICD->SNOMED links. The XML has entries like:
        <Description>
            <hasCode>icd9#412</hasCode>
            <Linked rdf:resource="http://snomed.info/id/212505002" />
        </Description>
    This function returns a list of dictionaries, each containing:
    
        {
            'icd_raw': 'icd9#412',
            'snomed_uri': 'http://snomed.info/id/212505002'
        }
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
    # 1) Load the ICD descriptions from CSV
    icd_descriptions = load_icd_descriptions(ICD_CSV)

    # 2) Build a dictionary of SNOMED concept -> label(s)
    snomed_labels = parse_snomed_turtle(SNOMED_TTL)

    # 3) Parse the XML file for ICD->SNOMED links
    links = parse_icd_snomed_links(KG)
    print(f"Found {len(links)} ICD->SNOMED links in the XML.\n")

    # 4) For each link, find the best matching SNOMED label and compute similarity.
    #    Finally, write the results to a CSV file for further review.
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        # Write header row
        writer.writerow([
            "icd_raw",
            "icd_extracted",
            "icd_text",
            "snomed_uri",
            "snomed_extracted",
            "best_snomed_label",
            "similarity",
            "verdict"
        ])

        for link in links:
            icd_raw_value = link["icd_raw"]
            snomed_uri_value = link["snomed_uri"]

            # Extract the ICD and SNOMED codes/IDs
            icd_code = extract_icd9_code(icd_raw_value)
            snomed_id = extract_snomed_id(snomed_uri_value)

            # Build ICD text by concatenating its short and long descriptions
            icd_text = ""
            if icd_code and icd_code in icd_descriptions:
                short_t = icd_descriptions[icd_code]["short"]
                long_t = icd_descriptions[icd_code]["long"]
                icd_text = f"{short_t} {long_t}".strip()

            # Gather all SNOMED labels associated with this concept
            candidate_labels = snomed_labels.get(snomed_id, []) if snomed_id else []

            # Find the label with the highest similarity to the ICD description
            best_similarity = 0.0
            best_label = ""
            normalized_icd = normalize(icd_text)
            for label in candidate_labels:
                normalized_label = normalize(label)
                sim_score = levenshtein_similarity(normalized_icd, normalized_label)
                if sim_score > best_similarity:
                    best_similarity = sim_score
                    best_label = label

            # If similarity passes the threshold, we mark it as "OK" else "Suspicious"
            verdict = "OK" if best_similarity >= SIM_THRESHOLD else "Suspicious"

            # Write everything to CSV
            writer.writerow([
                icd_raw_value,
                icd_code if icd_code else "",
                icd_text,
                snomed_uri_value,
                snomed_id if snomed_id else "",
                best_label,
                f"{best_similarity:.3f}",
                verdict
            ])

    print(f"Similarity results written to '{OUTPUT_CSV}'.\nDone.")


# if __name__ == "__main__":
#     main()