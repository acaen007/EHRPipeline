from SPARQLWrapper import SPARQLWrapper, JSON
import math

class Validator:
    def __init__(self, sparql_endpoint):
        self.sparql = SPARQLWrapper(sparql_endpoint)

    def execute_count_query(self, query):
        """
        Execute a SPARQL query that returns one row with something like ?count or ?sum.
        Return that integer value. Raise exception if the SPARQL endpoint says 'bad request'.
        """
        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()  # might raise QueryBadFormed
        if not results["results"]["bindings"]:
            return 0
        row = results["results"]["bindings"][0]
        for varname in ("count", "sum", "b1", "b2", "b3"):
            if varname in row:
                return int(row[varname]["value"])
        return 0

    def get_domain_range(self, predicate):
        """Get the rdfs:domain and rdfs:range of 'predicate'."""
        query = f"""
        SELECT ?domain ?range
        WHERE {{
          OPTIONAL {{ <{predicate}> rdfs:domain ?domain. }}
          OPTIONAL {{ <{predicate}> rdfs:range ?range. }}
        }}
        """
        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()
        if results["results"]["bindings"]:
            row = results["results"]["bindings"][0]
            dom = row.get("domain", {}).get("value")
            rng = row.get("range", {}).get("value")
            return dom, rng
        return None, None

    def count_instances(self, type_uri):
        """number of resources that are typed as <type_uri>."""
        if not type_uri:
            return 0
        query = f"""
        SELECT (COUNT(*) as ?count)
        WHERE {{
          ?x a <{type_uri}> .
        }}
        """
        return self.execute_count_query(query)

    def count_typed_predicate(self, predicate, domain, range_):
        """c_p: # of typed triples (?s, p, ?o) with ?s a <domain> and ?o a <range>."""
        if not domain or not range_:
            return 0
        query = f"""
        SELECT (COUNT(*) as ?count)
        WHERE {{
          ?s a <{domain}> .
          ?o a <{range_}> .
          ?s <{predicate}> ?o .
        }}
        """
        return self.execute_count_query(query)

    def build_nested_path_count_query(self, path, domain, range_, with_predicate=None):
        """
        Build a single SPARQL query that counts how many times a path "path" (list of predicates)
        occurs among typed resources (?s in domain, ?o in range_), using the nested approach
        from the paper.

        - path: [q1, q2, ..., qL]
        - domain: D(p)
        - range_: R(p)
        - with_predicate: None or the p to co-occur with (s, p, o)

        The output is a SPARQL query that yields one row with ?sum = Σ(Π_i ?b_i)
        (the sum of the product of the counts at each step).

        This matches the formal approach in the paper for counting:
        P(Π^L_{(tx, ty), q⃗})
        but implemented in a nested manner to avoid DISTINCT overhead.
        """
        L = len(path)
        if L == 0:
            return "SELECT (0 as ?sum) WHERE {}"

        # We'll define variable names for each node in the path: v0=?s, v1=?v1, ..., vL=?vL
        # b1..bL for aggregator vars
        v_nodes = ["?s"] + [f"?v{i}" for i in range(1, L)] + ["?o"]
        b_vars  = [f"?b{i}" for i in range(1, L+1)]

        # Step 1: inside subquery for the final triple
        def make_inner_subquery(p_L, v_subj, v_obj, range_iri, b_L):
            return f"""
    SELECT (COUNT({v_obj}) AS {b_L}) {v_subj} {v_obj}
    WHERE {{
    {v_subj} <{p_L}> {v_obj} .
    {v_obj} a <{range_iri}> .
    }}
    GROUP BY {v_subj} {v_obj}
    """.strip()

        # Step 2: each wrap outward
        def wrap_subquery(prev_subq, p_i, v_subj, v_obj, grouping_vars, b_i):
            group_by_str = " ".join(grouping_vars + [v_subj])
            return f"""
    SELECT (COUNT({v_obj}) AS {b_i}) {" ".join(grouping_vars)} {v_subj}
    WHERE {{
    {v_subj} <{p_i}> {v_obj} .
    {{
        {prev_subq}
    }}
    }}
    GROUP BY {group_by_str}
    """.strip()

        # Build from inside out
        p_L = path[L-1]
        v_subj_Lm1 = v_nodes[L-1]
        v_obj_L = v_nodes[L]
        b_L = b_vars[L-1]

        # innermost
        subq = make_inner_subquery(p_L, v_subj_Lm1, v_obj_L, range_, b_L)
        cur_subq = subq
        cur_grouping = [b_L, v_obj_L, v_subj_Lm1]

        # now wrap L-1 down to 1
        for step in reversed(range(1, L)):
            p_i = path[step-1]
            v_subj_i = v_nodes[step-1]
            v_obj_i   = v_nodes[step]
            b_i = b_vars[step-1]
            new_subq = wrap_subquery(
                prev_subq     = cur_subq,
                p_i           = p_i,
                v_subj        = v_subj_i,
                v_obj         = v_obj_i,
                grouping_vars = cur_grouping,
                b_i           = b_i
            )
            # update grouping
            next_group = [b_i]
            for c in cur_grouping:
                if c not in next_group:
                    next_group.append(c)
            if v_subj_i not in next_group:
                next_group.append(v_subj_i)
            cur_subq = new_subq
            cur_grouping = next_group

        # Insert domain check at the outermost level
        domain_injection = f"{v_nodes[0]} a <{domain}> ."
        lines = cur_subq.split("\n")
        replaced_subq_lines = []
        inserted = False
        for line in lines:
            if (not inserted) and ("WHERE {" in line):
                replaced_subq_lines.append(line)
                replaced_subq_lines.append(f"  {domain_injection}")
                inserted = True
            else:
                replaced_subq_lines.append(line)
        final_subq_with_domain = "\n".join(replaced_subq_lines)

        # Final top-level SELECT with optional co-occurring triple ?s <p> ?o
        product_expr = " * ".join(b_vars)
        top_query = f"""
    SELECT (SUM({product_expr}) AS ?sum)
    WHERE {{
    {{
        {final_subq_with_domain}
    }}
    {"?s <"+with_predicate+"> ?o ." if with_predicate else ""}
    }}
    """.strip()

        return top_query

    def count_typed_path_occurrences(self, path, domain, range_):
        """Count occurrences of a typed path."""
        if not domain or not range_:
            return 0
        query = self.build_nested_path_count_query(path, domain, range_, with_predicate=None)
        return self.execute_count_query(query)

    def count_typed_path_and_predicate_occurrences(self, path, predicate, domain, range_):
        """Count occurrences of a typed path with a predicate."""
        if not domain or not range_:
            return 0
        query = self.build_nested_path_count_query(path, domain, range_, with_predicate=predicate)
        return self.execute_count_query(query)

    def compute_npmi(self, c_path, c_path_p, c_p, c_domain, c_range):
        """Compute the normalized pointwise mutual information."""
        denom = c_domain * c_range
        if denom == 0:
            return 0.0
        p_path = c_path / denom
        p_path_p = c_path_p / denom
        p_p = c_p / denom

        if p_path == 0 or p_path_p == 0 or p_p == 0:
            return 0.0

        val = p_path_p / (p_path * p_p)
        if val <= 0:
            return 0.0
        num = math.log(val)
        den = -math.log(p_path_p)
        if den == 0:
            return 0.0
        npmi = num / den
        return max(npmi, 0.0)

    def discover_paths(self,subject, object_, max_length=3):
        """
        A local path discovery for length=1..max_length. 
        Returns lists of property URIs.  (We do NOT apply type constraints here.)
        This version enumerates *all* forward property chains up to length max_length.
        For each length L, we build a query:
        SELECT DISTINCT ?p1 ?p2 ... ?pL
        WHERE {
            <subject> ?p1 ?v1 .
            ?v1 ?p2 ?v2 .
            ...
            ?v(L-1) ?pL <object_> .
        }
        Then we read off each row as a path [p1, p2, ..., pL].
        """
        paths = []
        self.sparql.setReturnFormat(JSON)

        for length in range(1, max_length + 1):
            # Build the SELECT line: SELECT DISTINCT ?p1 ?p2 ... ?pL
            p_vars = " ".join([f"?p{i}" for i in range(1, length+1)])

            # Build the WHERE pattern:
            #   <subject> ?p1 ?v1 .
            #   ?v1 ?p2 ?v2 .
            #   ...
            #   ?v(L-1) ?pL <object_> .
            triples = []
            if length == 1:
                # single triple
                triples.append(f"<{subject}> ?p1 <{object_}> .")
            else:
                # chain
                triples.append(f"<{subject}> ?p1 ?v1 .")
                for i in range(2, length):
                    triples.append(f"?v{i-1} ?p{i} ?v{i} .")
                # final triple
                triples.append(f"?v{length-1} ?p{length} <{object_}> .")

            where_pattern = "\n  ".join(triples)
            q = f"""
    SELECT DISTINCT {p_vars}
    WHERE {{
    {where_pattern}
    }}
    """
            #print(f"\n--- Path discovery for length={length} ---\n{q}")
            self.sparql.setQuery(q)
            results = self.sparql.query().convert()

            # For each result row, we gather the path as an array
            for row in results["results"]["bindings"]:
                path_list = []
                for i in range(1, length+1):
                    path_list.append(row[f"p{i}"]["value"])
                paths.append(path_list)

        print("Discovered local paths:", paths)
        return paths


    def validate_fact(self, subject, predicate, object_, max_length=3):
        """
        Validate a fact using the COPAAL approach.
        """
        domain, range_ = self.get_domain_range(predicate)
        if not domain or not range_:
            print("No domain/range => can't run typed approach. Returning 0.")
            return 0.0

        c_domain = self.count_instances(domain)
        c_range = self.count_instances(range_)
        c_p = self.count_typed_predicate(predicate, domain, range_)

        if c_domain == 0 or c_range == 0:
            return 0.0

        paths = self.discover_paths(subject, object_, max_length=max_length)

        npmi_scores = []
        for path in paths:
            cpath = self.count_typed_path_occurrences(path, domain, range_)
            cpathp = self.count_typed_path_and_predicate_occurrences(path, predicate, domain, range_)

            if cpathp > 0:
                npmi_val = self.compute_npmi(cpath, cpathp, c_p, c_domain, c_range)
                print(f"Calculated npmi for Path={path}")
                npmi_scores.append(npmi_val)

        if not npmi_scores:
            return 0.0

        one_minus_vals = [1 - sc for sc in npmi_scores]
        product_ = math.prod(one_minus_vals)
        veracity = 1 - product_
        return veracity