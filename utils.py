def get_param_value(file_path, param_name):
    """Return the value of a parameter (key = 'value') from a text file."""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue  # skip comments or blank lines

            if line.startswith(param_name + " "):  # match start of line
                parts = line.split("=", 1)
                if len(parts) == 2:
                    value = parts[1].strip().strip("'\"")  # remove spaces and quotes
                    return value
    return None