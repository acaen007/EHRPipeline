from rdflib import Graph, Namespace, RDF, URIRef

# Adjust these as needed
INPUT_FILE = "CustomOntology.ttl"
OUTPUT_FILE = "enhanced_triples_FINAL_2.txt"
FORMATTED_OUTPUT_FILE = "formatted_triples_FINAL_2.txt"

# For convenience, define some known namespaces:
SPHN = Namespace("https://biomedit.ch/rdf/sphn-schema/sphn#")

# A small helper map to rewrite URI prefixes:
PREFIX_MAP = [
    ("http://example.org/Code/icd9#",    "icd9#"),
    ("http://example.org/Code/loinc#",   "loinc#"),
    ("http://snomed.info/id/",           "snomed#"),
    ("http://example.org/Diagnosis/",    "Diagnosis/"),
    ("http://example.org/LabResult/",    "LabResult/"),
    ("http://example.org/LabTestEvent/", "LabTestEvent/"),
    ("http://example.org/LabTest/",      "LabTest/"),
    ("http://example.org/PATIENTS/",     "PATIENTS/"),
    ("http://example.org/",              ""),  # fallback for local URIs
]

def rewrite_uri(uri_str):
    """
    Given a full URI string, rewrite it according to the PREFIX_MAP rules above.
    Returns a shortened string (e.g. 'icd9#51919' or 'Diagnosis/10033/PATIENTS/112572').
    """
    # Convert rdflib URIRef to plain string, just in case
    uri_str = str(uri_str)
    for long_prefix, short_prefix in PREFIX_MAP:
        if uri_str.startswith(long_prefix):
            return uri_str.replace(long_prefix, short_prefix, 1)
    return uri_str  # fallback if no match is found

def rewrite_type(uri_str):
    """
    If the object is a known SPHN class (e.g. sphn:Diagnosis, sphn:LabTest),
    rewrite to just 'Diagnosis', 'LabTest', etc. Otherwise, do our standard rewrite.
    """
    # You might check if it's in the SPHN namespace
    base = str(uri_str)
    if base.startswith(str(SPHN)):
        # local name
        return base.replace(str(SPHN), "")
    else:
        # if it's not in SPHN, apply the normal rewrite
        return rewrite_uri(uri_str)

def extract_triples():
    g = Graph()
    g.parse(INPUT_FILE, format='turtle')

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for s, p, o in g:
            # We only output lines if the subject is a URIRef (not a blank node).
            if not isinstance(s, URIRef):
                continue

            sub_str = rewrite_uri(s)

            if p == RDF.type:
                # Then the object is a class
                # e.g. LabTestEvent, Diagnosis, etc.
                pred_str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
                obj_str = rewrite_type(o)
            else:
                # For other predicates
                if str(p).startswith(str(SPHN)):
                    # e.g. "https://biomedit.ch/rdf/sphn-schema/sphn#hasCode"
                    # We keep only the local name, e.g. "hasCode"
                    pred_str = str(p).replace(str(SPHN), "")
                else:
                    # fallback or some other property
                    pred_str = rewrite_uri(p)

                # The object could be a URIRef or a Literal
                if isinstance(o, URIRef):
                    obj_str = rewrite_uri(o)
                else:
                    # If it's a literal (not in the example, but let's handle it)
                    # We'll just put it in quotes. Adjust if needed.
                    obj_str = f"\"{o}\""

            line = f"{sub_str} {pred_str} {obj_str}\n"
            out.write(line)

def format_triples():
    with open(OUTPUT_FILE, "r", encoding="utf-8") as infile, open(FORMATTED_OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for line in infile:
            parts = line.strip().split()
            if len(parts) >= 3:
                subject = parts[0]
                predicate = parts[1]
                obj = " ".join(parts[2:])
                outfile.write(f"{subject}\t{predicate}\t{obj}\n")

def main():
    extract_triples()
    format_triples()

if __name__ == "__main__":
    main()
