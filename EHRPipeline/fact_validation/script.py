from factValidation import Validator

def main():
    sparql_endpoint = "http://localhost:7200/repositories/finalrepohealthcare"
    validator = Validator(sparql_endpoint)

    predictions_file = "predictions.txt"
    output_file = "validated_facts.txt"
    
    with open(predictions_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue

            # Expect exactly 3 parts: subject, predicate, object
            parts = line.split()
            if len(parts) != 3:
                print(f"Skipping malformed line: {line}")
                continue
            
            subj = parts[0].strip()
            pred = parts[1].strip()
            obj  = parts[2].strip()

            subj_uri = subj.strip("<>")
            pred_uri = pred.strip("<>")
            obj_uri  = obj.strip("<>")

            # Validate
            score = validator.validate_fact(subj_uri, pred_uri, obj_uri, max_length=3)

            print(f"Fact: {subj} {pred} {obj} => Score: {score}")

            # threshold for writing the facts validated
            if score >= 0.5:
                f_out.write(f"{subj} {pred} {obj}\n")

    print(f"Validation complete. Facts with score >= 0.5 are in '{output_file}'.")

if __name__ == "__main__":
    main()
