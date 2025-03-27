import os

dossier = './'  

# Tous les fichiers d'extraction ont un nom commençant par "extraction"
fichiers_extraction = [f for f in os.listdir(dossier) if f.startswith("extraction") and f.endswith(".py")]

# Exécution
for fichier in fichiers_extraction:
    with open(fichier, 'r') as f:
        code = f.read()
        exec(code)