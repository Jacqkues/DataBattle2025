import torch
import time
from qdrant_client.http import models




class Retrivial:

    def __init__(self,colbert,processor,model,qdrant_client) -> None:
        self.processor = processor
        self.qdrant_client = qdrant_client
        self.model = model
        self.colbert = colbert


    def generate_embeding(self,query):
        with torch.no_grad():
            batch_query = self.processor.process_queries([query]).to(
                self.model.device
            )
            query_embedding = self.model(**batch_query)
        return query_embedding
    
    def fetch_question(self,query,k=5):
        return self.qdrant_client.query_points(
            collection_name="questions",
            query=list(self.colbert.query_embed(query))[0], #converting generator object into numpy.ndarray
            limit=5, #How many closest to the query movies we would like to get
            #with_vectors=True, #If this option is used, vectors will also be returned
            with_payload=True #So metadata is provided in the output
        )
    
    def fetch_doc_page(self,query,query_filter,k=3):
        embedding = self.generate_embeding(query)
        multivector_query = embedding[0].cpu().float().numpy().tolist()

        start_time = time.time()
        search_result = self.qdrant_client.query_points(
            collection_name="legal_basis_binary",
            query=multivector_query,
            limit=k,
            timeout=100,
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    ignore=False,
                    rescore=True,
                    oversampling=2.0,
                )
            ),
            query_filter=query_filter
        )
        end_time = time.time()        
        elapsed_time = end_time - start_time
        return (search_result,elapsed_time)