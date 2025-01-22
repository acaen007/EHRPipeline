# Neuro-Symbolic Toolkit for Clinical Data Management

Welcome to the Neuro-Symbolic Toolkit for Clinical Data Management! This repository provides a suite of tools for managing, validating, and analyzing clinical data using neuro-symbolic methods. The toolkit is designed to facilitate the alignment of clinical entities, validate facts, and ensure semantic completeness. It is especially tailored for data from clinical datasets such as MIMIC-III.

## Features
- **Entity Alignment**: Tools for aligning clinical entities across datasets.
- **Entity Linking**: Linking and validation of clinical concepts (e.g., ICD to SNOMED).
- **Fact Validation**: Methods for validating the consistency of clinical facts.
- **Schema Mapping**: Mapping tabular data to graph.
- **Semantic Completeness**: Embeddings generation of datasets.
- **MIMIC-III Integration**: Built-in support for the MIMIC-III dataset.

---

## Set up the Environment

To get started, follow these steps to set up the environment:

1. **Create a Python Virtual Environment:**
   ```bash
   python3 -m venv venv
   ```
2. **Activate the Virtual Environment:**
    - MacOS / Linux
      ```bash
      source venv/bin/activate
      ```
    - Windows
      ```bash
      venv\Scripts\activate
      ```
3. **Install Required Packages:**
    ```bash
    pip install -r requirements.txt 
    ```

## Repository Structure
EHRPIPELINE
```lua
|-- EHRPipeline
|   |-- entity_alignment/         # Tools for entity alignment
|   |-- entity_linking/           # Tools for entity linking
|       |-- icd_snomed_validation.csv  # Sample linking data
|       |-- linking_validation.py      # Validation scripts
|   |-- fact_validation/          # Methods for fact validation
|   |-- semantic_completeness/    # Scripts ensuring semantic completeness
|   |-- schema_mapping/           # Schema mapping files and scripts
|       |-- schema_mapping.py          # Mapping logic for ontologies
|       |-- D_ICD_DIAGNOSES.csv        # Example input file
|       |-- enhanced_sphn_triples_sample_FINAL.ttl  # Enhanced ontology triples
|   |-- mimic-iii/                # Placeholder for MIMIC-III dataset files
|
|-- Pipeline_UseCase1.ipynb       # Example use case in Jupyter Notebook
|-- .gitignore                    # Files and directories to ignore in Git
|-- README.md                     # Documentation for the repository
|-- requirements.txt              # List of Python dependencies
```

## How to Run

1. **Prepare Input Data:**
   Ensure you have your input data (e.g., CSV files, ontology files) ready in the appropriate directories (e.g., `schema_mapping/`, `entity_linking/`, `data/`).

2. **Run a Use Case:**
   Open and execute the example Jupyter Notebook `Pipeline_UseCase1.ipynb` to see the toolkit in action. You can use the notebook as a guide for running the individual components of the pipeline.

   To run the notebook:
   ```bash
   jupyter notebook Pipeline_UseCase1.ipynb
   ```

3. **Run Individual Scripts:**
   Each component of the pipeline can be executed individually. For example, to cross-ontology alignment:
   ```bash
   python EHRPipeline/entity_alignment/entity_alignment.py
   ```
   The `entity_alignment.py` script contains a class `CrossOntologyAligner` with two runnable methods:
   - `transcribe(self, query: rdflib.Graph, Invoker: str, Threshold=0.8, Namespace="http://example.org/sphn#")`
   - `merge(self, query: rdflib.Graph, Invoker: str, Threshold=0.8, Namespace="http://example.org/sphn#") -> rdflib.Graph`

   You can call these methods within a Python environment to perform specific alignment tasks.
---
