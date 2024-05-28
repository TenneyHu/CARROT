import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
emb = None

def load_model(modelName):
    global emb
    emb = SentenceTransformer(modelName)
    emb.to(device)

def get_retrieve_results(querys, index_file_path, english_recipes, modelName, results_num):
    global emb
    if emb is None:
        load_model(modelName)

    results = []
    query_vectors = []

    for line in querys:
        emb_zh = emb.encode(line, convert_to_tensor=True, device=device)
        emb_zh = emb_zh.unsqueeze(0) 
        query_vectors.append(emb_zh)

    query_vectors = [qv.cpu().numpy() for qv in query_vectors]
    query_vectors = np.vstack(query_vectors)
    faiss.normalize_L2(query_vectors)

    index = faiss.read_index(index_file_path)
    _, I = index.search(query_vectors, results_num)  
    
        
    for indices in I:
        result = []
        for _, index in enumerate(indices):
            try:
                result.append(english_recipes[index])
            except:
                pass
        results.append(result)

    return results