import os
import json
import re
import fitz  # PyMuPDF
from extraction_qcm_EPAC import fusion_tout_qcm

###############################
# Partie 1 : Extraction et parsing depuis les PDF EQE
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
    Nettoie le texte d'un bloc en supprimant les lignes de consigne et celles
    qui ne contiennent pas de véritables questions.
    """
    # Supprimer les lignes commençant par "For each of the statements" (insensible à la casse)
    cleaned = re.sub(r'(?im)^for each of the statements.*$', '', text)
    # Supprimer les lignes vides ou contenant uniquement un tiret
    cleaned_lines = []
    for line in cleaned.splitlines():
        ls = line.strip()
        if ls in ["–", "-"] or not ls:
            continue
        cleaned_lines.append(ls)
    return "\n".join(cleaned_lines)

def parse_questions_pdf(text):
    """
    Parse le PDF des questions pour EQE.
    Chaque bloc débute par "Question X" et contient :
      - "intitule" : le texte avant la première sous-question,
      - "subquestions" : une liste de sous-questions identifiées par le format "X.Y <texte>".
    """
    blocks = re.split(r'(?=Question\s+\d+)', text, flags=re.IGNORECASE)
    result = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        header_match = re.search(r'Question\s+(\d+)', block, re.IGNORECASE)
        if not header_match:
            continue
        major_q = header_match.group(1)
        # Conserver le contenu après "Question X"
        block_content = block.split('\n', 1)[-1].strip()
        block_content = clean_block_content(block_content)
        # Extraction des sous-questions au format "X.Y <texte>"
        subq_matches = re.findall(r'(\d+\.\d+)\s+(.*?)(?=\s+\d+\.\d+\s+|$)', block_content, re.DOTALL)
        if subq_matches:
            first_occurrence = re.search(r'(\d+\.\d+)\s+', block_content)
            intitule = block_content[:first_occurrence.start()].strip() if first_occurrence else ""
            subquestions = []
            for sub_number, question_text in subq_matches:
                question_text = " ".join(question_text.split())
                if question_text in ["–", "-"]:
                    continue
                subquestions.append({
                    "sub_number": sub_number.strip(),
                    "question_text": question_text
                })
        else:
            intitule = block_content
            subquestions = []
        result.append({
            "major_question": major_q,
            "intitule": intitule,
            "subquestions": subquestions
        })
    return result

def parse_responses_pdf(text):
    """
    Parse le PDF des réponses EQE pour extraire un dictionnaire.
    Pour chaque bloc débutant par "Question X", capture les sous-questions au format
    "X.Y – Answer" (où Answer est True/False).
    Le résultat est un dictionnaire indexé par le numéro de la grande question.
    """
    blocks = re.split(r'(?=Question\s+\d+)', text, flags=re.IGNORECASE)
    result = {}
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        header_match = re.search(r'Question\s+(\d+)', block, re.IGNORECASE)
        if not header_match:
            continue
        major_q = header_match.group(1)
        subq_pattern = re.compile(r'(\d+\.\d+)\s*[–-]\s*(True|False)', re.IGNORECASE)
        subq_matches = subq_pattern.findall(block)
        subquestions = []
        for num, ans in subq_matches:
            subquestions.append({
                "sub_number": num.strip(),
                "answer": ans.capitalize()
            })
        result[major_q] = {
            "response_text": block,
            "subquestions": subquestions
        }
    return result

def merge_questions_responses(questions, responses):
    """
    Fusionne les données issues des PDF questions et réponses EQE.
    Pour chaque grande question, ajoute le champ "answer_general" (issu de "response_text")
    et associe à chaque sous-question la réponse correspondante, si trouvée.
    Le champ "type" est forcé à "mcq" (pour EQE, QCM).
    """
    merged = []
    for q in questions:
        major = q.get("major_question")
        merged_block = {
            "type": "mcq",  # Pour EQE, on fixe le type à "mcq"
            "title": f"Question {major}",
            "intitule": q.get("intitule"),
            "subquestions": q.get("subquestions", []),
            "answer_general": ""
        }
        resp_block = responses.get(major, {})
        merged_block["answer_general"] = resp_block.get("response_text", "")
        merged.append(merged_block)
    return merged

def fill_empty_subanswers(merged_data):
    """
    Pour chaque bloc, si une sous-question n'a pas de réponse (champ "answer" vide),
    extrait les paires "x: True/False" depuis "answer_general" et met à jour les sous-questions.
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
# Partie 2 : Transformation vers le JSON final (sans fichiers intermédiaires)
###############################

def process_item(item, filename):
    """
    Pour un bloc issu du merge, fusionne l'intitulé et les sous-questions pour créer le champ "question".
    Chaque sous-question est préfixée par son numéro.
    Le champ "answer" est repris depuis "answer_general".
    Les autres champs (title, intitule, subquestions) sont supprimés.
    """
    base_text = item.get("intitule", "").strip()
    subq_list = item.get("subquestions", [])
    subq_texts = []
    for sub in subq_list:
        num = sub.get("sub_number", "").strip()
        text = sub.get("question_text", "").strip()
        # On préfixe la sous-question par son numéro
        if num:
            subq_texts.append(f"{num}: {text}")
        else:
            subq_texts.append(text)
    if subq_texts:
        merged_question = base_text + "\n\n" + "\n".join(subq_texts) if base_text else "\n".join(subq_texts)
    else:
        merged_question = base_text

    new_item = {
        "file": os.path.basename(filename),
        "type": item.get("type", "mcq"),
        "question": merged_question,
        "answer": item.get("answer_general", "").strip()
    }
    return new_item

def transform_to_final_format(merged_data, filename):
    """
    Applique process_item à chaque bloc et enveloppe le tout dans un dictionnaire avec la clé "content".
    """
    final_items = [process_item(item, filename) for item in merged_data]
    return {"content": final_items}

def clean_questions(data):
    for item in data.get("content", []):
        if "Jan Feb" in item["question"]:
            item["question"] = item["question"].split("Jan Feb")[0].strip()
        if "JANUARY FEBRUARY MARCH" in item["question"]:
            item["question"] = item["question"].split("JANUARY FEBRUARY MARCH")[0].strip()
        
    data["content"] = [item for item in data.get("content", []) if item["question"] != "For"]
            
    return data



###############################
# Partie 3 : Traitement complet pour EQE sans fichiers intermédiaires
###############################

def process_eqe_pdf(questions_pdf, responses_pdf, output_json):
    """
    Traite un couple de fichiers PDF EQE (questions et réponses) et écrit le JSON final.
    """
    # Extraction des textes depuis les PDF
    questions_text = extract_text_from_pdf(questions_pdf)
    responses_text = extract_text_from_pdf(responses_pdf)
    
    if not questions_text:
        print(f"Aucun texte extrait du PDF questions : {questions_pdf}")
        return
    if not responses_text:
        print(f"Aucun texte extrait du PDF réponses : {responses_pdf}")
        return

    # Parsing des textes
    questions_data = parse_questions_pdf(questions_text)
    responses_data = parse_responses_pdf(responses_text)
    
    # Fusion des données questions et réponses
    merged_data = merge_questions_responses(questions_data, responses_data)
    merged_data = fill_empty_subanswers(merged_data)
    
    # Transformation vers le format final
    final_json = transform_to_final_format(merged_data, questions_pdf)
    
    final_json = clean_questions(final_json)
    
    # Écriture du JSON final
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final_json, f, ensure_ascii=False, indent=4)
    print(f"JSON final généré : {output_json}")
    


###############################
# Partie 4 : Traitement de tous les fichiers EQE dans un dossier
###############################


def process_all_eqe():
    """
    Parcourt le dossier 'script/avant_EQE' pour regrouper par année les fichiers questions et réponses EQE,
    puis pour chaque année, traite le couple questions/réponses et génère un fichier JSON final dans
    le dossier 'json_EQE_qcm'.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "avant_EQE")
    output_dir = os.path.join(script_dir, "json_EQE_qcm")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # On parcourt tous les fichiers dans le dossier d'entrée
    files = os.listdir(input_dir)
    
    # On regroupe par année : on recherche par exemple "2012_PreEx_questions" et "2012_PreEx_answers"
    grouped = {}  # {année: { "questions": chemin, "answers": chemin } }
    pattern = re.compile(r'(\d{4})_PreEx_.*(questions|answers)', re.IGNORECASE)
    for file in files:
        match = pattern.match(file)
        if match:
            year = match.group(1)
            typ = match.group(2).lower()
            if year not in grouped:
                grouped[year] = {}
            grouped[year][typ] = os.path.join(input_dir, file)
    
    # Pour chaque année qui possède à la fois questions et réponses, lancer le traitement
    for year, paths in grouped.items():
        if "questions" in paths and "answers" in paths:
            output_json = os.path.join(output_dir, f"{year}_merged.json")
            print(f"Traitement de l'année {year} avec questions: {paths['questions']} et réponses: {paths['answers']}")
            process_eqe_pdf(paths["questions"], paths["answers"], output_json)
        else:
            print(f"Année {year} : fichier questions ou réponses manquant.")

###############################
# Partie 5 : Main
###############################

if __name__ == '__main__':
    process_all_eqe()
    
    dossier = "./json_EQE_qcm"
    fichiers = ["2012_merged.json", "2013_merged.json", "2014_merged.json", "2015_merged.json", "2016_merged.json", "2017_merged.json"]
        
    fusion = fusion_tout_qcm(fichiers , dossier)
    