import pandas as pd
import random
import re
import rdflib
import urllib.parse

FRACTION = 0.05  # 5%

PATIENTS_CSV = "PATIENTS.csv"
DIAGNOSES_ICD_CSV = "DIAGNOSES_ICD.csv"
LABEVENTS_CSV = "LABEVENTS.csv"
DLABITEMS_CSV = "D_LABITEMS.csv"

OUTPUT_TTL = "enhanced_sphn_triples_sample_FINAL.ttl"

random.seed(42)

patients_df = pd.read_csv(PATIENTS_CSV)
diagnoses_icd_df = pd.read_csv(DIAGNOSES_ICD_CSV)
lab_events_df = pd.read_csv(LABEVENTS_CSV)
lab_items_df = pd.read_csv(DLABITEMS_CSV)

all_subjects = patients_df['subject_id'].unique().tolist()
random.shuffle(all_subjects)

num_5pct = int(len(all_subjects) * FRACTION)
keep_subjects = set(all_subjects[:num_5pct])

patients_df_small = patients_df[patients_df['subject_id'].isin(keep_subjects)]
diagnoses_icd_df_small = diagnoses_icd_df[diagnoses_icd_df['subject_id'].isin(keep_subjects)]
lab_events_df_small = lab_events_df[lab_events_df['subject_id'].isin(keep_subjects)]

keep_itemids = set(lab_events_df_small['itemid'].dropna().unique())
lab_items_df_small = lab_items_df[lab_items_df['itemid'].isin(keep_itemids)]

itemid_to_loinc = {}
for _, row in lab_items_df_small.iterrows():
    itemid = row['itemid']
    loinc = str(row['loinc_code']).strip()
    if loinc == 'nan' or loinc == '':
        loinc = None
    itemid_to_loinc[itemid] = loinc

def sanitize_value_for_iri(value):
    """Sanitize values for use in IRI format."""
    if pd.isna(value):
        return "NA"
    val_str = str(value)
    return urllib.parse.quote(val_str, safe="-._~")

triples_ttl = []

for idx, row in diagnoses_icd_df_small.iterrows():
    row_id = row['row_id']
    subj_id = row['subject_id']
    icd9_code = str(row['icd9_code']).strip()

    diagnosis_iri = f"<http://example.org/Diagnosis/{subj_id}/PATIENTS/{row_id}>"
    subject_pseudo_iri = f"<http://example.org/PATIENTS/{subj_id}>"
    icd9_iri = f"<http://example.org/Code/icd9#{icd9_code}>"

    triples_ttl.append(f"{diagnosis_iri} a sphn:Diagnosis .")
    triples_ttl.append(f"{subject_pseudo_iri} a sphn:SubjectPseudoIdentifier .")
    triples_ttl.append(f"{diagnosis_iri} sphn:hasSubjectPseudoIdentifier {subject_pseudo_iri} .")
    triples_ttl.append(f"{icd9_iri} a sphn:Code .")
    triples_ttl.append(f"{diagnosis_iri} sphn:hasCode {icd9_iri} .")

for idx, row in lab_events_df_small.iterrows():
    row_id = row['row_id']
    subj_id = row['subject_id']
    itemid = row['itemid']
    val = row['value']

    if pd.isna(row_id) or pd.isna(subj_id) or pd.isna(itemid):
        continue

    lab_event_iri = f"<http://example.org/LabTestEvent/{int(subj_id)}/PATIENTS/{int(row_id)}>"
    subject_pseudo_iri = f"<http://example.org/PATIENTS/{int(subj_id)}>"
    lab_test_iri = f"<http://example.org/LabTest/{int(subj_id)}/PATIENTS/{int(itemid)}>"
    
    value_part = sanitize_value_for_iri(val)
    lab_result_iri = f"<http://example.org/LabResult/{int(subj_id)}/PATIENTS/{int(itemid)}/{value_part}>"

    triples_ttl.append(f"{lab_event_iri} a sphn:LabTestEvent .")
    triples_ttl.append(f"{subject_pseudo_iri} a sphn:SubjectPseudoIdentifier .")
    triples_ttl.append(f"{lab_event_iri} sphn:hasSubjectPseudoIdentifier {subject_pseudo_iri} .")

    triples_ttl.append(f"{lab_test_iri} a sphn:LabTest .")
    triples_ttl.append(f"{lab_event_iri} sphn:hasLabTest {lab_test_iri} .")

    triples_ttl.append(f"{lab_result_iri} a sphn:LabResult .")
    triples_ttl.append(f"{lab_test_iri} sphn:hasResult {lab_result_iri} .")

    loinc_code = itemid_to_loinc.get(itemid, None)
    if loinc_code is not None:
        loinc_iri = f"<http://example.org/Code/loinc#{loinc_code}>"
        triples_ttl.append(f"{loinc_iri} a sphn:Code .")
        triples_ttl.append(f"{lab_test_iri} sphn:hasCode {loinc_iri} .")
        triples_ttl.append(f"{lab_result_iri} sphn:hasCode {loinc_iri} .")

with open(OUTPUT_TTL, "w", encoding="utf-8") as f:
    for line in triples_ttl:
        f.write(line)
        if not line.endswith("\n"):
            f.write("\n")

print(f"Sample of the data has been transformed into '{OUTPUT_TTL}' with {len(triples_ttl)} RDF statements.")

def merge_ontology_and_data():
    """
    Merges the base SPHN ontology snippet from a file (initialsphn.ttl)
    with additional data (enhanced_triples_sample.ttl) into a single graph.
    """
    g = rdflib.Graph()

    g.parse("initialsphn.ttl", format="turtle")

    g.parse("enhanced_sphn_triples_sample_FINAL.ttl", format="turtle")

    g.serialize(destination="merged_ontology.ttl", format="turtle")

    print("Merging completed! Saved as 'merged_ontology.ttl'.")

  
