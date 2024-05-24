import faiss
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def build_index(document_file_path, index_file_path, modelName, dimension):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inputs = []
    with open(document_file_path, 'r') as f:
        for line in f:
            if len(line.strip().split('\t')) > 1:
                inputs.append(line.strip().split('\t')[1])
            else :
                inputs.append(line.strip())

    batch_size = 128
    index = faiss.IndexFlatIP(dimension)

    with torch.no_grad():
        emb = SentenceTransformer(modelName)
        emb.to(device)
        num_batches = (len(inputs) + batch_size - 1) // batch_size
        
        for i in tqdm(range(num_batches), desc="Encoding"):

            batch_inputs = inputs[i*batch_size:(i+1)*batch_size]
            emb_en = emb.encode(batch_inputs, convert_to_tensor=True, batch_size=batch_size)
            for vector in emb_en:
                vector = vector.cpu().numpy().reshape(1, -1)
                faiss.normalize_L2(vector)
                index.add(vector)

    faiss.write_index(index, index_file_path)