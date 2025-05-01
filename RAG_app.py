import faiss                                     # For building a vector similarity search index
import numpy as np                               # For handling embeddings as NumPy arrays
import torch                                     # Needed for device handling in transformers
from transformers import pipeline                # HuggingFace pipeline for generation
from sentence_transformers import SentenceTransformer  # For embedding text into vectors
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For chunking text

# --- 3.1 Suppress noisy logs and warnings ---
import logging
from transformers.utils import logging as hf_logging
import warnings

# Reduce logging noise from libraries
logging.getLogger("langchain.text_splitter").setLevel(logging.ERROR)
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# --- 3.2 Parameters ---
chunk_size = 500                  # Character length of each chunk
chunk_overlap = 50                # Number of overlapping characters between chunks
model_name = "sentence-transformers/all-distilroberta-v1"  # SentenceTransformer to use
top_k = 5                         # How many chunks to retrieve for each query

# --- 3.3 Read the pre-scraped document ---
with open("Selected_Document.txt", "r", encoding="utf-8") as f:
    text = f.read()

# --- 3.4 Split into chunks ---
text_splitter = RecursiveCharacterTextSplitter(
    separators=['\n\n', '\n', ' ', ''],          # Chunk splitting priority
    chunk_size=chunk_size,                      # Max chunk size
    chunk_overlap=chunk_overlap                 # Overlap between chunks
)
chunks = text_splitter.split_text(text)         # Create the list of text chunks

# --- 3.5 Embed & Build FAISS Index ---
print("Embedding text and building FAISS index...")
embedder = SentenceTransformer(model_name)      # Load the embedding model
embeddings = embedder.encode(                   # Encode chunks to vector form
    chunks, convert_to_numpy=True, show_progress_bar=False
).astype('float32')

dimension = embeddings.shape[1]                 # Get the embedding vector size
index = faiss.IndexFlatL2(dimension)            # Create a brute-force FAISS index (L2 distance)
index.add(embeddings)                           # Add all chunk vectors to the index

# --- 3.6 Load the generator pipeline ---
generator = pipeline("text2text-generation", model="google/flan-t5-small", device=-1)
# device=-1 means run on CPU; use device=0 for GPU

# --- 3.7 Retrieval and answering functions ---
def retrieve_chunks(question, k=top_k):
    # Convert question to embedding
    query_embedding = embedder.encode([question], convert_to_numpy=True).astype('float32')
    # Search top-k similar chunks
    distances, indices = index.search(query_embedding, k)
    # Return the top-k text chunks
    return [chunks[i] for i in indices[0]]

def answer_question(question):
    # Get relevant context chunks
    context = "\n\n".join(retrieve_chunks(question))
    # Build prompt using retrieved context
    prompt = f"Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    # Generate answer using text-to-text model
    result = generator(prompt, max_new_tokens=200)
    # Return only the answer portion
    return result[0]['generated_text'].split("Answer:")[-1].strip()

# --- 3.8 Interactive input loop ---
if __name__ == "__main__":
    print("Enter 'exit' or 'quit' to end.")
    while True:
        question = input("Your question: ")
        if question.lower() in ("exit", "quit"):
            break
        print("Answer:", answer_question(question))
