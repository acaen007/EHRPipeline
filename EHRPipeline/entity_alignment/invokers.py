import pandas as pd
import logging
import rdflib
import abc
import re

###############################################################################
#                              Abstract invoker class                         #
###############################################################################
class Invoker(abc.ABC):
    def remoteInvoker(self, func_name: str, *args, **kwargs):
        """
        Dynamically invokes a method
        """
        if not hasattr(self, func_name):
            raise AttributeError(f"The function '{func_name}' does not exist in the invoker.")

        func = getattr(self, func_name)

        if not callable(func):
            raise TypeError(f"'{func_name}' is not callable.")
        return func(*args, **kwargs)

    def loadModule(self, id: str) -> pd.DataFrame:
        paths = { # add ID and file path to the mapping 
            "icd9": "../data/D_ICD_DIAGNOSES.csv",
            "labresults" : "data/D_LABITEMS.csv"
        }
        if id not in paths:
            raise ValueError(f"Module ID '{id}' not found in paths.")

        try:
            dataframe = pd.read_csv(paths[id])
        except FileNotFoundError:
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")
        except pd.errors.EmptyDataError:
            raise ValueError(f"The file '{file_path}' is empty or corrupted.")
        except Exception as e:
            raise RuntimeError(f"An error occurred while loading the file '{file_path}': {e}")
        return dataframe

###############################################################################
#                     Custom Invokers for graphs                              #
###############################################################################
class GraphInvoker(Invoker):
    def __init__(self, embeddingModel: object, namespace) -> None:
        self.embeddingModel = embeddingModel
        self.namespace = namespace

    def icd9tosnomed(self, query: rdflib.Graph) -> rdflib.Graph:
        module = self.loadModule("icd9")
        modifiedGraph = rdflib.Graph()
        logging.debug("Loaded module: ICD9 data")

        for subject, predicate, obj in query.triples((None, self.namespace, None)):
            logging.debug(f"Processing triple: {subject}, {predicate}, {obj}")

            if not isinstance(obj, rdflib.URIRef):
                logging.warning(f"Skipping non-URI object: {obj}")
                continue

            match = re.search(r"icd9#(.*)$", str(obj))
            if not match:
                logging.warning(f"ICD9 code not found in object: {obj}")
                continue

            code = match.group(1)
            logging.debug(f"Extracted ICD9 code: {code}")

            matched_row = module[module["ICD9_CODE"] == code]
            if matched_row.empty:
                logging.warning(f"No match found for ICD9 code in DataFrame: {code}")
                continue

            long_title = matched_row.iloc[0]["LONG_TITLE"]
            logging.debug(f"Found long title: {long_title}")

            try:
                embedding = self.embeddingModel.encode(long_title, show_progress_bar=False)
                logging.debug(f"Generated embedding for code {code}")
            except Exception as e:
                logging.error(f"Error generating embedding for {long_title}: {e}")
                continue

            modifiedGraph.add((subject, self.namespace, rdflib.Literal(embedding)))
            logging.debug(f"Added triple to graph: ({subject}, {self.namespace}, embedding)")

        if len(modifiedGraph) == 0:
            logging.error("Modified graph is empty: No ICD codes were processed!")
            raise ValueError("No ICD codes found in the input graph.")

        logging.debug("Returning modified graph")
        return modifiedGraph

    def icd9tosnomed_old(self, query: rdflib.Graph) -> rdflib.Graph:
        module = self.loadModule("icd9")
        modifiedGraph = rdflib.Graph()

        for subject, predicate, obj in query.triples((None, self.namespace, None)):
            if not isinstance(obj, rdflib.URIRef):
                continue
            match = re.search(r"icd9#(.*)$", str(obj))
            if not match:
                continue  
            code = match.group(1)

            matched_row = module[module["ICD9_CODE"] == code]
            if matched_row.empty:
                continue
            long_title = matched_row.iloc[0]["LONG_TITLE"]

            embedding = self.embeddingModel.encode(long_title, show_progress_bar=False)

            modifiedGraph.add((subject, self.namespace, rdflib.Literal(embedding)))
        return modifiedGraph

    def labresults2snomed(self, query: rdflib.Graph) -> rdflib.Graph:
        module = self.loadModule("labresults")
        modifiedGraph = rdflib.Graph()

        for subject, predicated, obj in query.triples((None, self.namespace, None)):
            if isinstance(obj, rdflib.URIRef):
                match = re.search(r"/(\d+)$", str(obj))
                if not match:
                    continue
                code = match.group(1)

                matched_row = module[module["ITEMID"].astype(str) == code]
                if matched_row.empty:
                    continue
                description = matched_row.iloc[0]["LABEL"]

                embedding = self.embeddingModel.encode(description, show_progress_bar=False)

                modifiedGraph.add((subject, self.namespace, rdflib.Literal(embedding)))
        return modifiedGraph

    def snomedtoicd9(self, query: rdflib.Graph) -> rdflib.Graph:
        modifiedGraph = rdflib.Graph()

        for subject, predicate, obj in query.triples((None, self.namespace, None)):
            if isinstance(obj, rdflib.URIRef):
                snomed_url = str(obj)
                embedding = self.embeddingModel.encode(snomed_url, show_progress_bar=False)
                modifiedGraph.add((subject, self.namespace, rdflib.Literal(embedding)))

        return modifiedGraph
