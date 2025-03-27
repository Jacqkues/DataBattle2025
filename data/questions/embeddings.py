import json
from qdrant_client import QdrantClient, models
from fastembed import LateInteractionTextEmbedding
import uuid

def lecture_qcm(path):
    
    """
    Calcule la surface d'un rectangle.

    Args:
        path (string): Chemin du fichier.

    Returns:
        float: La surface du rectangle.
    """
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def embedding(questions, model):
    return list(
        model.embed(questions)
    )
  
def recup_questions(data):
  return [item["question"] for item in data["content"]]  
  
def lecture_dossier_json_epac_qcm(relative_path, fichiers):
    
    #Ouverture des json séparés et du fichier fusionné
    data = lecture_qcm(relative_path + fichiers[0])
    data2 = lecture_qcm(relative_path + fichiers[1])
    data3 = lecture_qcm(relative_path + fichiers[2])
    data4 = lecture_qcm(relative_path + fichiers[3])
        
    # Récupération des questions pour embedding
    questions_2022 = recup_questions(data)
    questions_2023 = recup_questions(data2)
    questions_2024 = recup_questions(data3)
    questions_mock = recup_questions(data4)
    
    return questions_2022, questions_2023, questions_2024, questions_mock, data, data2, data3, data4

def lecture_dossier_json_eqe_qcm(relative_path, fichiers):
    
    #Ouverture des json séparés et du fichier fusionné
    data = lecture_qcm(relative_path + fichiers[0])
    data2 = lecture_qcm(relative_path + fichiers[1])
    data3 = lecture_qcm(relative_path + fichiers[2])
    data4 = lecture_qcm(relative_path + fichiers[3])
    data5 = lecture_qcm(relative_path + fichiers[4])
    data6 = lecture_qcm(relative_path + fichiers[5])
        
    # Récupération des questions pour embedding
    questions_2012 = recup_questions(data)
    questions_2013 = recup_questions(data2)
    questions_2014 = recup_questions(data3)
    questions_2015 = recup_questions(data4)
    questions_2016 = recup_questions(data5)
    questions_2017 = recup_questions(data6)
    
    return questions_2012, questions_2013, questions_2014, questions_2015, questions_2016, questions_2017, data, data2, data3, data4, data5, data6

def lecture_dossier_open(relative_path):

    open_1 = lecture_qcm(relative_path_open + "2022_merged_final.json")
    open_2 = lecture_qcm(relative_path_open + "2023_merged_final.json")
    open_3 = lecture_qcm(relative_path_open + "2024_merged_final.json")
    
    open_total = lecture_qcm(relative_path + "QCM.json")

    
    questions_2022 = recup_questions(open_1)
    questions_2023 = recup_questions(open_2)
    questions_2024 = recup_questions(open_3)
    
    return questions_2022, questions_2023, questions_2024, open_total

def creation_collection(nom):
    #suppression
    qdrant_client.delete_collection(collection_name=nom)
    
    #creation
    qdrant_client.delete_collection(collection_name=nom)
    qdrant_client.create_collection(
        collection_name=nom,
        vectors_config=models.VectorParams(
            size=128, #size of each vector produced by ColBERT
            distance=models.Distance.COSINE, #similarity metric between each vector
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM #similarity metric between multivectors (matrices)
            ),
        ),
    )  
    
def sauvegarde_bdd(collection_name, embeddings, total_json):
    qdrant_client.upload_points(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=str(uuid.uuid4()),
                payload=total_json["content"][idx],
                vector=vector
            )
            for idx, vector in enumerate(embeddings)
        ],
    )


def filtrer_questions(json_obj, max_length=2000):
    """
    Supprime les éléments de 'content' où la clé 'question' dépasse max_length caractères.
    """
    if "content" in json_obj and isinstance(json_obj["content"], list):
        # Filtrer les éléments dont la question est trop longue
        json_obj["content"] = [item for item in json_obj["content"] if len(item.get("question", "")) <= max_length]

    return json_obj
       
def fusionner_json_qcm(liste):
    """
    Fusionne une liste de JSON en combinant les listes sous la clé 'content'.
    """
    nouveau = {"content": []}

    for json_obj in liste:
        if "content" in json_obj and isinstance(json_obj["content"], list):
            nouveau["content"].extend(json_obj["content"])  # Ajoute les éléments à la liste

    return nouveau

if __name__ == "__main__":
    
    # Récupération des données json

    # QCM EPAC
    relative_path = "./json_EPAC_qcm/"
    fichiers = ["2022_qcm.json", "2023_qcm.json", "2024_qcm.json", "mock_qcm.json"]
    questions_2022_epac, questions_2023_epac, questions_2024_epac, questions_mock_epac, json_2022_epac, json_2023_epac, json_2024_epac, json_mock_epac = lecture_dossier_json_epac_qcm(relative_path, fichiers)
    json_total_epac = fusionner_json_qcm([json_2022_epac, json_2023_epac, json_2024_epac, json_mock_epac])
    
    #QCM EQE
    relative_path = "./json_EQE_qcm/"
    fichiers = ["2012_merged.json", "2013_merged.json", "2014_merged.json", "2015_merged.json", "2016_merged.json", "2017_merged.json"]
    questions_2012_eqe, questions_2013_eqe, questions_2014_eqe, questions_2015_eqe, questions_2016_eqe, questions_2017_eqe, json_eqe_2012, json_eqe_2013, json_eqe_2014, json_eqe_2015, json_eqe_2016, json_eqe_2017 = lecture_dossier_json_eqe_qcm(relative_path, fichiers)
    
    questions_2013_eqe = [item for item in questions_2013_eqe if len(item) <= 2000]
    questions_2012_eqe = [item for item in questions_2012_eqe if len(item) <= 2000]
    questions_2014_eqe = [item for item in questions_2014_eqe if len(item) <= 2000]
    questions_2015_eqe = [item for item in questions_2015_eqe if len(item) <= 2000]
    questions_2016_eqe = [item for item in questions_2016_eqe if len(item) <= 2000]
    questions_2017_eqe = [item for item in questions_2017_eqe if len(item) <= 2000]
    
    json_eqe_2012 = filtrer_questions(json_eqe_2012)
    json_eqe_2013 = filtrer_questions(json_eqe_2013)
    json_eqe_2014 = filtrer_questions(json_eqe_2014)
    json_eqe_2015 = filtrer_questions(json_eqe_2015)
    json_eqe_2016 = filtrer_questions(json_eqe_2016)
    json_eqe_2017 = filtrer_questions(json_eqe_2017)
    
    total_qcm_eqe = fusionner_json_qcm([json_eqe_2012, json_eqe_2013, json_eqe_2014, json_eqe_2015, json_eqe_2016, json_eqe_2017])

    # Questions ouvertes
    relative_path_open = "./json_EPAC_open/"
    questions_open_2022, questions_open_2023, questions_open_2024, total_open = lecture_dossier_open(relative_path_open)
    

    # # # Colbert
    em_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")

    # Connexion à la bdd
    qdrant_client = QdrantClient(
        url="https://d1faa86f-45a0-49bc-8080-4e5b6ea1703a.eu-west-1-0.aws.cloud.qdrant.io",
        api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.cSE45xYWHgS9q_rbwf3186l925E7OovQEgBz6s7x_uc",
        timeout=300
    )
    
    #creation des collection
    creation_collection("questions")

    #embeddings
    embeddings_total_open = embedding(questions_open_2022, em_model) + embedding(questions_open_2023, em_model) + embedding(questions_open_2024, em_model)
    embedding_qcm_epac = embedding(questions_2022_epac, em_model) + embedding(questions_2023_epac, em_model) + embedding(questions_2024_epac, em_model) + embedding(questions_mock_epac, em_model)
    embeddings_2012_eqe = embedding(questions_2012_eqe, em_model)
    embeddings_2013_eqe = embedding(questions_2013_eqe, em_model)
    embeddings_2014_eqe = embedding(questions_2014_eqe, em_model)
    embeddings_2015_eqe = embedding(questions_2015_eqe, em_model)
    embeddings_2016_eqe = embedding(questions_2016_eqe, em_model)
    embeddings_2017_eqe = embedding(questions_2017_eqe, em_model)

    #open
    sauvegarde_bdd("questions", embeddings_total_open, total_open)   
    sauvegarde_bdd("questions", embedding_qcm_epac, json_total_epac)
    sauvegarde_bdd("questions", embeddings_2012_eqe, json_eqe_2012)
    sauvegarde_bdd("questions", embeddings_2013_eqe, json_eqe_2013)
    sauvegarde_bdd("questions", embeddings_2014_eqe, json_eqe_2014)
    sauvegarde_bdd("questions", embeddings_2015_eqe, json_eqe_2015)
    sauvegarde_bdd("questions", embeddings_2016_eqe, json_eqe_2016)
    sauvegarde_bdd("questions", embeddings_2017_eqe, json_eqe_2017)
