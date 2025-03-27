import os
import json
import re
import fitz  # PyMuPDF
from extraction_qcm_EPAC import fusion_tout_qcm


###############################
# Partie 1 : Extraction et Parsing depuis les PDF EPAC
###############################

def extract_text_from_pdf(pdf_path):
    """Extrait l'intégralité du texte d'un PDF avec PyMuPDF."""
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text("text") + "\n"
    except Exception as e:
        print(f"Erreur lors de l'extraction du texte de {pdf_path} : {e}")
    return text

def clean_block_content(text):
    """
    Supprime les lignes de consigne (ex : "For each of the statements") et les lignes vides.
    """
    cleaned = re.sub(r'(?im)^for each of the statements.*$', '', text)
    lines = []
    for line in cleaned.splitlines():
        ls = line.strip()
        if ls in ["–", "-"] or not ls:
            continue
        lines.append(ls)
    return "\n".join(lines)

def parse_epac_questions(text):
    """
    Parse le PDF des questions EPAC.
    Chaque bloc débute par une ligne de la forme "X. (Y points)".
    Dans chaque bloc :
      - L'intitulé est le texte situé après le header et avant la première sous-question.
      - Les sous-questions débutent par un marqueur sous la forme "a. ", "b. ", etc.
    Retourne une liste de dictionnaires contenant :
      - "block_number" : le numéro du bloc (ex : "2022" si le header commence par "2022. (…)")
      - "intitule" : le texte d'intitulé
      - "subquestions" : une liste de sous-questions avec "sub_number" et "question_text"
    """
    # Découpage des blocs à partir d'un header de type "X. (… points)"
    block_pattern = re.compile(r"^(?P<block>\d+)\.\s*\(.*points\)", re.MULTILINE)
    splits = block_pattern.split(text)
    blocks = []
    it = iter(splits)
    pre = next(it, "")  # Texte avant le premier header (souvent vide)
    while True:
        num = next(it, None)
        if num is None:
            break
        content = next(it, "")
        blocks.append({"block_number": num.strip(), "content": content.strip()})
    
    parsed_blocks = []
    # Pattern pour détecter les sous-questions débutant par "a. ", "b. ", etc.
    subq_pattern = re.compile(r"^(?P<num>[a-z])\.\s*(?P<text>.+)$", re.IGNORECASE | re.MULTILINE)
    for block in blocks:
        content = block["content"]
        subq_matches = list(subq_pattern.finditer(content))
        if subq_matches:
            first_subq_pos = subq_matches[0].start()
            intitule = content[:first_subq_pos].strip()
            subs = []
            for match in subq_matches:
                subs.append({
                    "sub_number": match.group("num").strip(),
                    "question_text": match.group("text").strip()
                })
        else:
            intitule = content.strip()
            subs = []
        parsed_blocks.append({
            "block_number": block["block_number"],
            "intitule": intitule,
            "subquestions": subs
        })
    return parsed_blocks

def parse_responses_pdf(text):
    """
    Parse le PDF des réponses pour extraire un dictionnaire.
    Pour chaque bloc débutant par "Question X" (en début de ligne), on capture le numéro
    et tout le texte qui suit.
    Retourne un dictionnaire { "X": <texte complet de la réponse> }.
    """
    pattern = re.compile(r"^Question\s+(\d+)", re.IGNORECASE | re.MULTILINE)
    splits = pattern.split(text)
    responses = {}
    it = iter(splits)
    pre = next(it, "")
    while True:
        num = next(it, None)
        if num is None:
            break
        resp = next(it, "").strip()
        responses[num.strip()] = resp
    return responses

def merge_questions_responses(questions, responses):
    """
    Fusionne les données issues des PDF questions et réponses EPAC.
    Pour chaque bloc de questions, on ajoute le champ "answer_general" à partir
    du dictionnaire responses (en se basant sur "block_number").
    """
    merged = []
    for q in questions:
        num = q.get("block_number")
        merged_block = {
            "type": "Open",  # Pour EPAC, le type est "Open"
            "intitule": q.get("intitule"),
            "subquestions": q.get("subquestions", []),
            "answer_general": responses.get(num, "")
        }
        merged.append(merged_block)
    return merged

def fill_empty_subanswers(merged_data):
    """
    Pour chaque bloc, si une sous-question n'a pas de réponse renseignée,
    essaie de l'extraire depuis "answer_general" en recherchant le pattern "a: True" par exemple.
    """
    pattern = re.compile(r"([a-z])\s*:\s*(True|False)", re.IGNORECASE)
    for block in merged_data:
        ans_text = block.get("answer_general", "")
        mapping = {}
        for match in pattern.finditer(ans_text):
            mapping[match.group(1).lower()] = match.group(2).capitalize()
        for subq in block.get("subquestions", []):
            if not subq.get("answer"):
                sub_num = subq.get("sub_number", "").lower()
                if sub_num in mapping:
                    subq["answer"] = mapping[sub_num]
    return merged_data

###############################
# Partie 2 : Transformation vers le JSON final
###############################

def process_item(item, filename):
    """
    Pour un bloc issu du merge, fusionne l'intitulé et les sous-questions pour créer le champ "question".
    Chaque sous-question est préfixée par son numéro.
    Le champ "answer" est repris depuis "answer_general".
    Les autres champs (block_number, intitule, subquestions) sont supprimés.
    """
    base_text = item.get("intitule", "").strip()
    subq_list = item.get("subquestions", [])
    subq_texts = []
    for sub in subq_list:
        num = sub.get("sub_number", "").strip()
        text = sub.get("question_text", "").strip()
        subq_texts.append(f"{num}: {text}")
    if subq_texts:
        merged_question = base_text + "\n\n" + "\n".join(subq_texts) if base_text else "\n".join(subq_texts)
    else:
        merged_question = base_text

    new_item = {
        "file": os.path.basename(filename),
        "type": item.get("type", "Open"),
        "question": merged_question,
        "answer": item.get("answer_general", "").strip()
    }
    return new_item

def transform_to_final_format(merged_data, filename):
    """
    Applique process_item sur chaque bloc et enveloppe le tout dans {"content": [...]}
    """
    final_items = [process_item(item, filename) for item in merged_data]
    return {"content": final_items}

###############################
# Partie 3 : Traitement complet depuis les PDF EPAC (sans fichiers intermédiaires)
###############################

def process_pdf_pair(questions_file, responses_file, output_file):
    """
    Traite une paire de fichiers PDF (questions et réponses) pour EPAC et écrit le JSON final.
    """
    questions_text = extract_text_from_pdf(questions_file)
    responses_text = extract_text_from_pdf(responses_file)
    
    if not questions_text or not responses_text:
        print(f"Erreur : impossible d'extraire le texte depuis {questions_file} ou {responses_file}.")
        return

    # Parsing des questions et réponses
    questions_data = parse_epac_questions(questions_text)
    responses_data = parse_responses_pdf(responses_text)
    
    # Fusion et remplissage des réponses manquantes
    merged_data = merge_questions_responses(questions_data, responses_data)
    merged_data = fill_empty_subanswers(merged_data)
    
    # Transformation finale vers le format JSON souhaité
    final_json = transform_to_final_format(merged_data, questions_file)
    
    # Écriture du JSON final
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_json, f, ensure_ascii=False, indent=4)
    print(f"JSON généré : {output_file}")

def process_all_documents(input_dir, output_dir):
    """
    Parcourt le répertoire input_dir (contenant les PDF EPAC) et regroupe par année
    les fichiers questions et réponses.
    Pour chaque année, s'il existe à la fois un fichier exam et un fichier solution,
    le traitement est lancé et le JSON final est écrit dans output_dir.
    
    On s'attend à ce que les noms de fichiers contiennent l'année et un identifiant :
      - Pour les questions : un nom contenant "EPAC_exam" (par exemple, "2022 - EPAC_exam_open_en.pdf")
      - Pour les réponses : un nom contenant "EPAC_solution" (par exemple, "2022 - EPAC_solution_mcq&open-7-9.pdf")
    """
    # Dictionnaire pour regrouper par année
    groups = {}
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(".pdf"):
            continue
        # Utilisation d'une regex pour extraire l'année et l'identifiant ("exam" ou "solution")
        match = re.search(r"(\d{4})\s*-\s*EPAC_(exam|solution)", filename, re.IGNORECASE)
        if match:
            year = match.group(1)
            typ = match.group(2).lower()  # "exam" ou "solution"
            if year not in groups:
                groups[year] = {}
            groups[year][typ] = os.path.join(input_dir, filename)
    
    # Pour chaque année, si l'on a à la fois exam et solution, traiter la paire
    for year, files in groups.items():
        if "exam" in files and "solution" in files:
            output_file = os.path.join(output_dir, f"{year}_merged_final.json")
            print(f"Traitement de l'année {year}...")
            process_pdf_pair(files["exam"], files["solution"], output_file)
        else:
            print(f"Année {year} : fichier 'exam' ou 'solution' manquant.")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Répertoire contenant les PDF EPAC (questions et réponses)
    input_dir = os.path.join(script_dir, "EPAC_open")
    # Répertoire de sortie pour les JSON finaux
    output_dir = os.path.join(script_dir, "json_EPAC_open")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    process_all_documents(input_dir, output_dir)

if __name__ == '__main__':
    main()
    dossier = "./json_EPAC_open"
    fichiers = ["2022_merged_final.json", "2023_merged_final.json", "2024_merged_final.json"]

        
    fusion_tout_qcm(fichiers , dossier)
