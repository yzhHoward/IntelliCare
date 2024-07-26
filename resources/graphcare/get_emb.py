import requests
import json

# with open("../../resources/openai.key", 'r') as f:
#     key = f.readlines()[0][:-1]

# def embedding_retriever(term):
#     # Set up the API endpoint URL and request headers
#     url = "https://api.openai.com/v1/embeddings"
#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": f"Bearer {key}"
#     }

#     # Set up the request payload with the text string to embed and the model to use
#     payload = {
#         "input": term,
#         "model": "text-embedding-ada-002"
#     }

#     # Send the request and retrieve the response
#     response = requests.post(url, headers=headers, data=json.dumps(payload))

#     # Extract the text embeddings from the response JSON
#     embedding = response.json()["data"][0]['embedding']

#     return embedding


import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

revision = None  # Replace with the specific revision to ensure reproducibility if the model is updated.

# model = SentenceTransformer("/home/yzh/llm/frame_new/gte-large-en-v1.5/", revision=revision)
tokenizer = AutoTokenizer.from_pretrained("/home/yzh/llm/frame_new/gte-large-en-v1.5/", trust_remote_code=True)
model = AutoModel.from_pretrained("/home/yzh/llm/frame_new/gte-large-en-v1.5/", trust_remote_code=True).cuda()

texts = [
    "Illustration of the REaLTabFormer model. The left block shows the non-relational tabular data model using GPT-2 with a causal LM head. In contrast, the right block shows how a relational dataset's child table is modeled using a sequence-to-sequence (Seq2Seq) model. The Seq2Seq model uses the observations in the parent table to condition the generation of the observations in the child table. The trained GPT-2 model on the parent table, with weights frozen, is also used as the encoder in the Seq2Seq model.",
    "Predicting human mobility holds significant practical value, with applications ranging from enhancing disaster risk planning to simulating epidemic spread. In this paper, we present the GeoFormer, a decoder-only transformer model adapted from the GPT architecture to forecast human mobility.",
    "As the economies of Southeast Asia continue adopting digital technologies, policy makers increasingly ask how to prepare the workforce for emerging labor demands. However, little is known about the skills that workers need to adapt to these changes"
]

# Compute embeddings
# embeddings = model.encode(texts, convert_to_tensor=True)

def embedding_retriever(term):
    # embeddings = model.encode([term], convert_to_tensor=True)[0]
    input_ids = tokenizer.encode(term, padding=True, truncation=True, max_length=8192)
    with torch.no_grad():
        embeddings = model(torch.tensor(input_ids, device='cuda').unsqueeze(0))[0][0, 0]
    return embeddings.cpu().numpy()
