from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from services.retrivial import Retrivial
import os
import json
from openai import OpenAI
from utils import encode_img
import re
from services.genai import text_image_to_text

# Define your collection name
collection_name = 'questions'  # Change this to your collection name

client = QdrantClient(
        url="https://d1faa86f-45a0-49bc-8080-4e5b6ea1703a.eu-west-1-0.aws.cloud.qdrant.io",
        api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.cSE45xYWHgS9q_rbwf3186l925E7OovQEgBz6s7x_uc",
    )

colqwen_model = ColQwen2_5.from_pretrained(
        "vidore/colqwen2.5-v0.2",
    ).eval()
processor = ColQwen2_5_Processor.from_pretrained("vidore/colqwen2.5-v0.1")

retrivial = Retrivial(processor,colqwen_model,client)
base_image_path = "/data/PROJET_PERSO/pi/pi_backend/data/pdf-images"

def save_to_json(out, filename="output.json"):
    """Save the output dictionary to a JSON file."""
    with open(filename, 'w') as json_file:
        json.dump(out, json_file, indent=4)
    print(f"Saved {len(out)} items to {filename}.")

out = {}
def fetch_all_items_from_collection():
        # Fetch all items (vectors) from the collection
        # Assuming you want to fetch all vectors without any specific filtering.
        response = client.scroll(collection_name=collection_name,limit=200)
        print(len(response[0]))
        for item in response[0]:
            a = item.payload.get('answer')
            first_page = retrivial.fetch_doc_page(a,None,1)[0].points[0].payload
            print("fetched page ok")
            image_path  = os.path.join(base_image_path, first_page["pdf"], first_page["page"])

            out[a] = image_path
        save_to_json(out)

def load_json(filename="output.json"):
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
    
def append_to_file(filename, text):
    with open(filename, "a", encoding="utf-8") as file:
        file.write(text + "\n")

def prompt_resp():
    total = 0
    open_router_api = OpenAI(base_url="https://openrouter.ai/api/v1",api_key="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")
    data = load_json("./output.json")
    for key in data:
        image_path = data[key]
        prompt ="""

        # GOAL 
        Check for possible hallucination


        In  this image : <image>
        And in this answer
        ANSWER

        Check If the answer site the legal articles in the image output : <score>1</score>
        Else if the answer don't site the legals articles in the image output : <score>0</score>

        Strictly output in the <score> tags don't add any additional information !
        """
        prompt = prompt.replace("ANSWER",key)
        img = encode_img(image_path)
        try:
            result = text_image_to_text(open_router_api,prompt,img,"google/gemma-3-27b-it:free")
            pattern = re.compile(r"<score>\s*(.*?)\s*</score>\s*",
                re.DOTALL  # Allows matching across multiple lines
                )
            print("--------------------------")
            print(result)
            match = pattern.search(result)
            if match:
                s = match.group(1).strip()
                print(s)
                total += round(float(s))
                if round(float(s)) == 0:
                    append_to_file("./error_question",key)

            else:
                print(result)
        except:
            print("failed")
        
    print(total)


if __name__ == '__main__':
     prompt_resp()
    #fetch_all_items_from_collection()