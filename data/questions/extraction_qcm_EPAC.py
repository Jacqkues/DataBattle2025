import re
import json
import fitz
import os
from pathlib import Path
import glob


########## Extraction des questions dans des fichiers json ##############

# Conversion du pdf vers du texte
def lecture_pdf(pdf, relative_path, diviser=0):
    doc = fitz.open(relative_path+ '/'+ pdf)
    
    text = ""
    for page in doc:
        page_text = page.get_text()

        page_text = re.sub(r'\b\d+\n', '', page_text) 

        text += page_text
        
    if diviser == 1:
        part_index = text.find("Part 2")
        if part_index != -1:
            text = text[:part_index]
        
    return text
        
def extraction_questions(text, fichier, e=0):
    
    pattern = r"\d+\..*?(?:A\..*?B\..*?C\..*?D\..*?)(?=\s*\d+\.|\Z)"

    questions = re.findall(pattern, text, re.DOTALL)
    resultat = {"content": [{"question": q.strip(), "file": fichier} for q in questions]}
    
    if(e == 1):
        resultat = correct_json(resultat, 1)
    elif(e == 2):
        resultat = correct_json(resultat, 2)
    elif(e ==3):
        resultat = correct_json(resultat, 3)
        
    return resultat

def extraction_reponses_mock(text):
        
    question_regex = re.compile(r"Question (\d+)")
    answer_regex = re.compile(r"([A-D])")

    questions_and_answers = []

    matches = question_regex.findall(text)
    answers = answer_regex.findall(text)

    for match, an in zip(matches, answers):
        question_number = match
        questions_and_answers.append({
            "question": question_number,
            "answer": an
        })

    data = {"content": questions_and_answers}

    return data

def extraction_reponses(text):
    pattern = r"Question (\d+):\s*([A-D])\s*(.*?)(?=\nQuestion \d+:|\Z)"

    # Recherche des correspondances
    matches = re.findall(pattern, text, re.DOTALL)
    
    result = {"content": []}
    for match in matches:
        question, answer, justification = match
        result["content"].append({
            "question": question.strip(),
            "answer": answer.strip() + ". " + justification.strip()
        })

    return result
           
def fusion_questions_reponses(data1, data2, nom, dossier):   
    if not os.path.exists(dossier):
        os.makedirs(dossier)

    answers_dict = {str(item["question"]): item for item in data2["content"]}

    merged_content = []
    for item in data1["content"]:
        full_question = item["question"]
        question_num = full_question.split(".")[0].strip()
        if question_num in answers_dict:
            merged_item = item.copy()
            merged_item.update(answers_dict[question_num])
            merged_item["type"] = "mcq"
            merged_item["question"] = full_question
            merged_content.append(merged_item)
        else:
            merged_content.append(item)

    merged_data = {"content": merged_content}
    chemin_fichier = os.path.join(dossier, f"{nom}_qcm.json")
    
    with open(chemin_fichier, "w", encoding="utf-8") as f_out:
        json.dump(merged_data, f_out, indent=4, ensure_ascii=False)

def fusion_tout_qcm(fichiers,dossier):

    if not os.path.exists(dossier):
        os.makedirs(dossier)
        
    qcm_total = {"content": []}

    for fichier in fichiers:
        with open(dossier + "/" + fichier, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "content" in data:
                qcm_total["content"].extend(data["content"])
              
    chemin_fichier = os.path.join(dossier, f"QCM.json")

    with open(chemin_fichier, "w", encoding="utf-8") as f:
        json.dump(qcm_total, f, indent=4, ensure_ascii=False)
        
    return qcm_total

def correct_json(data, e):
    questions = data["content"]
    
    if(e == 1):
        # Trouver l'e dans la question 14
        misplaced_text = "2023."
        for i, item in enumerate(questions):
            if item["question"].startswith("2023."):

                questions[i]["question"] = questions[i]["question"].replace(misplaced_text, "", 1).strip()
                

        # Ajouter "2023." à la fin de la question 13 si nécessaire
        for i, item in enumerate(questions):
            if item["question"].startswith("13."):
                questions[i]["question"] += " 2023"
        
    if(e == 2):
        misplaced_text = "2.0, patent search via publication server, \nMailbox"
        for i, item in enumerate(questions):
            if item["question"].startswith("2.0,"):
                questions[i]["question"] = questions[i]["question"].replace(misplaced_text, "", 1).strip()
                

        # Ajouter "2023." à la fin de la question 13 si nécessaire
        for i, item in enumerate(questions):
            if item["question"].startswith("11."):
                questions[i]["question"] += " " + misplaced_text

    if(e == 3):
        misplaced_text = "2022."
        for i, item in enumerate(questions):
            if item["question"].startswith(misplaced_text):
                questions[i]["question"] = questions[i]["question"].replace(misplaced_text, "", 1).strip()
                

        # Ajouter "2023." à la fin de la question 13 si nécessaire
        for i, item in enumerate(questions):
            if item["question"].startswith("5.") or item["question"].startswith("1."):
                questions[i]["question"] += " " + misplaced_text              
        
    return data


if __name__ == "__main__":
    
    relative_path = "./EPAC Exams"
    liste_qcm_epac = ["2022 - EPAC_exam_mcq_en.pdf", "2023 - EPAC_exam_MCQ_EN.pdf", "2024 - EPAC_exam_mcq.pdf", "MOCK2 - mock_mcq_en.pdf"]
    liste_reponse_qcm_epac = ["2022 - EPAC_Solution_mcq&open.pdf", "2023 - EPAC_solution_mcq.pdf", "2024 - EPAC_solution_mcq.pdf", "MOCK2 - mock_mcq_solutions_en.pdf"]
    dossier = "json_EPAC_qcm"

    q_2022 = lecture_pdf(liste_qcm_epac[0], relative_path)
    q_2023 = lecture_pdf(liste_qcm_epac[1], relative_path)
    q_2024 = lecture_pdf(liste_qcm_epac[2], relative_path)
    q_mock = lecture_pdf(liste_qcm_epac[3], relative_path)
    
    questions_2024 = extraction_questions(q_2024, liste_qcm_epac[2],2)
    questions_mock = extraction_questions(q_mock, liste_qcm_epac[3],3)
    questions_2022 = extraction_questions(q_2022, liste_qcm_epac[0])
    questions_2023 = extraction_questions(q_2023, liste_qcm_epac[1],1)
        
    
    ## Récupération des réponses uniquement
    r_2022 = lecture_pdf(liste_reponse_qcm_epac[0], relative_path, 1)
    r_2023 = lecture_pdf(liste_reponse_qcm_epac[1], relative_path)
    r_2024 = lecture_pdf(liste_reponse_qcm_epac[2], relative_path)
    r_mock = lecture_pdf(liste_reponse_qcm_epac[3], relative_path)
    
    reponses_mock = extraction_reponses_mock(r_mock)
    reponses_2022 = extraction_reponses(r_2022)
    reponses_2023 = extraction_reponses(r_2023)
    reponses_2024 = extraction_reponses(r_2024)
    
    fusion_questions_reponses(questions_2023, reponses_2023, "2023", dossier)
    fusion_questions_reponses(questions_2022, reponses_2022, "2022", dossier)
    fusion_questions_reponses(questions_2024, reponses_2024, "2024", dossier)
    fusion_questions_reponses(questions_mock, reponses_mock, "mock", dossier)

    # Fusion de tous les qcm avec leurs réponses dans un seul fichier
    def get_filenames(directory):
        return [f.name for f in Path(directory).iterdir() if f.is_file()]

    files = get_filenames(dossier)
    fusion_tout_qcm(files, dossier)
    

