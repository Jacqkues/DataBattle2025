import os
import torch
import time
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
import uuid
import stamina
from PIL import Image



qdrant_client = QdrantClient(
    url=os.getenv("DATABASE_URL"),
    api_key=os.getenv("DATABASE_KEY"),
)

colqwen_model = ColQwen2_5.from_pretrained(
        "vidore/colqwen2.5-v0.2",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",  # or "mps" if on Apple Silicon
    ).eval()
processor = ColQwen2_5_Processor.from_pretrained("vidore/colqwen2.5-v0.1")

collection_name = "legal_basis_binary"


qdrant_client.create_collection(
    collection_name=collection_name,
    on_disk_payload=True,  # store the payload on disk
    vectors_config=models.VectorParams(
        size=128,
        distance=models.Distance.COSINE,
        on_disk=True, # move original vectors to disk
        multivector_config=models.MultiVectorConfig(
            comparator=models.MultiVectorComparator.MAX_SIM
        ),
        quantization_config=models.BinaryQuantization(
        binary=models.BinaryQuantizationConfig(
            always_ram=True  # keep only quantized vectors in RAM
            ),
        ),
    ),
)

@stamina.retry(on=Exception, attempts=3) # retry mechanism if an exception occurs during the operation
def upsert_to_qdrant(points):
    try:
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points,
            wait=False,
        )
    except Exception as e:
        print(f"Error during upsert: {e}")
        return False
    return True

from PIL import Image

def process_files_in_folders(root_folder):
    for subdir, _, files in os.walk(root_folder):
        parent_name = os.path.basename(subdir)
        if parent_name != os.path.basename(root_folder):  # Exclude the root folder name
            for file in files:
                file_path = os.path.join(subdir, file)
                image = Image.open(file_path)
                with torch.no_grad():
                    batch_images = processor.process_images([image]).to(colqwen_model.device)
                    image_embeddings = colqwen_model(**batch_images)

                points = []
                for j, embedding in enumerate(image_embeddings):
                    # Convert the embedding to a list of vectors
                    multivector = embedding.cpu().float().numpy().tolist()
                    
                    points.append(
                        models.PointStruct(
                            id=str(uuid.uuid4()),  # we just use the index as the ID
                            vector=multivector,  # This is now a list of vectors
                            payload={
                               "pdf":parent_name,
                               "page":file
                            },  # can also add other metadata/data
                        )
                    )
                    
                try:
                    upsert_to_qdrant(points)
                    print(f"added {parent_name} : {file}")
                except Exception as e:
                    print(f"Error during upsert: {e}")
                    continue
                


                
process_files_in_folders("/kaggle/working/out1")
