import base64
import json
def encode_img(image_path):
    with open(image_path,"rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    
    return "data:image/png;base64," + encoded_string


def load_json(filename):
    """Read and load the JSON file into a Python dictionary."""
    try:
        with open(filename, 'r') as json_file:
            data = json.load(json_file)
            print(f"Loaded {len(data)} items from {filename}.")
            return data
    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: The file {filename} is not a valid JSON file.")
        return {}