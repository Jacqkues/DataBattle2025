import os


with open("./creation_json.py", 'r') as f:
    code = f.read()
exec(code)

with open("./embeddings.py", 'r') as f:
    code = f.read()
exec(code)