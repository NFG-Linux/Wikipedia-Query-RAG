I chose to use a wiki page about the video game called Rocket League. It is a game similar to soccer, except that the user is controlling a car that can jump and fly with the use of rockets.


Improvements that could be made:
🚀 1. Speed Improvements
✅ Faster Embedding

    Use smaller SentenceTransformer models (e.g., all-MiniLM-L6-v2, 384D)
    → Faster inference with minimal quality drop.

    Batch embeddings manually (if not already done by encode()).

    Run on GPU if available (device=0), especially for larger batches.

✅ Faster Generation

    Use smaller models (e.g., flan-t5-small vs flan-t5-base)

    Switch to quantized versions (e.g., ggml, int8) if using libraries like Optimum or onnxruntime.

🧠 2. Memory Efficiency
✅ Embedding Management

    Reduce chunk overlap if it's too high — overlap increases total embedding count.

    Store and reuse embeddings and the FAISS index using:

    faiss.write_index(index, "index.faiss")

    to avoid recomputing every run.

✅ Use Approximate Search for Large Datasets

    Switch from IndexFlatL2 to:

        IndexIVFFlat (inverted index, faster lookup)

        IndexHNSWFlat (graph-based, fast & accurate)

🧰 3. Functional Improvements
✅ Modular Config

    Externalize config via .yaml, .json, or command-line args (argparse) so you don’t have to modify source files.

✅ Multi-Source Document Ingestion

    Add support for .pdf, .docx, and .md in addition to URLs or .txt.

✅ Persistent Cache

    Cache results of common queries (e.g., using diskcache or sqlite) for instant re-answers.

✅ Asynchronous User Loop

    Use asyncio or threading for better UI responsiveness if deploying as a web app or CLI with multiple users.

📊 4. Evaluation & Debugging
✅ Logging

    Add timestamps and retrieval diagnostics to a log file.

    Show top-k retrieved chunks before generation.

✅ Evaluation Tools

    Integrate rouge, bertscore, or precision@k to automatically score performance during testing.

✨ Optional UX Enhancements

    Add a simple CLI menu or gradio/streamlit web interface

    Highlight which chunks were retrieved and used in the answer

    Provide feedback options (👍 / 👎) for future tuning


The AI tool I coordinated with outputted pretty great data. I read the top portion of the assignment and then started feeding it with data, but as I got further down the FSO page, I had to have it re-do a lot of the coding to fit the prompts and it handled the recoding expertly and with 0 problems. The RAG system output was interesting; I would need to tune the output before releasing this. I think the wording of my questions wasn't in line with how it was searching for data; especially for the awards and review questions; no matter which chunk values I used, those 2 questions always gave terrible answers. The initial extractor script only pulled the first main section, not the subsequent sections; didn't take long working with the AI tool to get the script pulling all the data from the website!


4.1
Chunk_size:
I changed the chunk size to be very small and the answers weren't relevant to what I asked, and it took longer to output the answers. When I made it a large chunk, the answerd were even less relevant. It took longer to build the FAISS index.
Chunk_overlap:
I changed the chunk overlap to a small number and the answers returned quicker and more accurate. A high overlap provided less precise answers overall, but the question about the objective was the only time it had anything remotely close. One game mode's objective is to hit the ball on the opponents side of the field to damage it.

4.2
You said:
How does the embedding model's dimensionality affect FAISS indexing and retrieval performance?
ChatGPT said:

Great question — embedding dimensionality plays a central role in how FAISS handles both retrieval accuracy and performance (speed/memory). Here's a detailed breakdown:
🧠 What Is Embedding Dimensionality?

When you encode a chunk of text using a SentenceTransformer model (like all-MiniLM-L6-v2), the result is a vector — typically of 384, 512, 768, or more dimensions. This is the dimensionality of the embedding space.
⚙️ How It Affects FAISS
✅ 1. Retrieval Accuracy

    Higher dimensionality = potentially richer semantic representation.
    More dimensions allow the model to encode finer-grained meaning, which can improve the precision of matching.

    But: diminishing returns — going from 384 → 768 may help, but 768 → 1024+ often gives marginal gains unless you’re working with very complex, nuanced documents.

🚫 2. Speed and Memory Usage

    FAISS uses more memory and CPU/GPU resources as dimensionality increases:

        Each vector is larger.

        Every search involves computing more distances per query.

        Large indexes (millions of vectors) scale poorly with high dimensions unless optimized.

    For IndexFlatL2 (what you're using), which is brute-force L2 search:

        Search time scales linearly with the number of vectors × embedding dimensionality.

        Index size = #chunks × dimension × 4 bytes (float32)

⚠️ 3. Index Type Sensitivity

    Advanced FAISS index types (like IVF, HNSW, PQ) are more sensitive to dimensionality — higher dimensions may hurt recall unless you tweak parameters carefully.

    Lower dimensions are easier to approximate effectively in those cases.

🧪 Example: SentenceTransformer Models
Model	Dimensionality	Speed (↑ = faster)	Quality (↑ = better)
all-MiniLM-L6-v2	384	🔹🔹🔹🔹🔹	🔹🔹🔹🔹
all-distilroberta-v1	768	🔹🔹🔹	🔹🔹🔹🔹🔹
all-mpnet-base-v2	768	🔹🔹🔹	🔹🔹🔹🔹🔹🔹
✅ TL;DR

    Use 384–768D for most RAG apps — great balance of quality and performance.

    Higher dimensions = better semantic nuance, but slower indexing/search and more RAM usage.

    Stick with IndexFlatL2 for small/medium apps; consider approximate indexes for huge corpora.

    When experimenting with new embedding models, always check the dimensionality to avoid mismatches in FAISS.

Want help switching to a higher-dimensional model and benchmarking it against your current one?
You said:
What search algorithm does FAISS use, and how does that impact result quality and speed?
ChatGPT said:

Excellent — understanding FAISS search algorithms is key to balancing speed and accuracy in your RAG system.
🔍 What Search Algorithm Does FAISS Use?

FAISS supports multiple search/index types, and the one you choose defines the algorithmic behavior. You're currently using:

faiss.IndexFlatL2

This uses the brute-force L2 (Euclidean distance) search algorithm.
⚙️ What Does IndexFlatL2 Do?

    Exhaustive search — it compares the query vector to every vector in the index.

    Uses L2 distance (a.k.a. Euclidean distance):
    L2(x,y)=∑i=1n(xi−yi)2
    L2(x,y)=i=1∑n​(xi​−yi​)2

    ​

    Returns the k closest vectors to your query.

✅ Result Quality & Speed Impact (for IndexFlatL2)
Feature	Impact
✅ Accuracy	Perfect — always finds the true top-k most similar chunks.
❌ Speed	Slower for large datasets (thousands to millions of vectors).
❌ Scalability	Grows linearly with dataset size — can hit memory and performance limits.
✅ Simplicity	Easy to use, no parameters to tune or train.

For your current use case (~hundreds or low-thousands of chunks), IndexFlatL2 is great: high accuracy, no setup cost.
🧠 Other FAISS Search Algorithms (for large-scale RAG)

If your dataset grows, you might want to use approximate indexes like:
Index Type	Description	Trade-off
IndexIVFFlat	Inverted file system with clustering + flat search inside cells	Much faster, needs training
IndexHNSWFlat	Graph-based (Hierarchical Navigable Small World graph) search	Fast & accurate, more complex
IndexPQ	Product quantization for ultra-compressed embeddings	Super memory-efficient, lower accuracy
IndexLSH	Locality-sensitive hashing (good for binary vectors)	Fast, but rough approximation

These introduce faster, scalable approximate search, but you lose 100% accuracy in exchange for big performance gains.
✅ TL;DR for Your RAG App

    IndexFlatL2 = accurate and simple, but gets slow as your chunk count grows.

    It’s ideal for prototyping and small-to-mid datasets.

    For production-scale RAG (>50k+ chunks), switch to something like IndexIVFFlat or IndexHNSWFlat and evaluate recall vs. speed.

Would you like help benchmarking your app using a different FAISS index type (like HNSW or IVF)?
You said:
How do we determine the optimal number of chunks to retrieve for accurate answer generation?
ChatGPT said:

Excellent — the top_k value (number of chunks retrieved) is one of the most important hyperparameters in your RAG system. It directly affects the context you give the language model and the quality of its answers.
🔍 What Is top_k?

It defines how many of the most relevant chunks (based on embedding similarity) are returned from FAISS for a given query:

def retrieve_chunks(question, k=top_k):
    ...

These k chunks are then joined into a prompt and passed to the generator model (like flan-t5).
⚖️ How top_k Affects Answer Quality
top_k Value	Pros	Cons
Low (1–3)	Faster, focused context	May miss key info or lack depth
Medium (4–7)	Balanced context and speed	Best for most questions
High (8–12+)	Broad coverage, great recall	Slower, risk of diluting relevance or exceeding model input size
🧠 Key Considerations for Picking top_k
✅ 1. Model’s Token Limit

    flan-t5-small: ~512–1024 tokens

    flan-t5-base: ~1024 tokens

    If your chunks are ~500 characters, you usually fit 4–6 chunks max before truncation happens.

    🛑 Too many chunks = LLM cuts off the end — and that might include the actual answer.

✅ 2. Question Type

    Specific factual question: 1–3 chunks may be enough (e.g., "When was Rocket League released?")

    Open-ended or multi-part question: You’ll want 5–7+ chunks (e.g., "Describe the development history and gameplay changes of Rocket League")

✅ 3. Redundancy & Retrieval Noise

    Top-k > 7 starts pulling in less relevant chunks, which may introduce noise or distract the model from the most useful content.

🧪 Practical Tuning Strategy

    Start with top_k = 5 — it balances context depth and performance.

    Try top_k = 3 for precision-focused questions.

    Benchmark with top_k = 8–10 only if your generator model can handle long prompts and you're asking broad questions.

    Print/log the retrieved chunks for analysis — are they too similar? Too irrelevant? Adjust accordingly.

✅ TL;DR

    top_k = 4–6 is the sweet spot for most RAG systems using chunk sizes of 400–600 characters.

    Tune it based on:

        Model token limits

        Question complexity

        Chunk size and overlap

Want me to add a command-line flag or input prompt so you can test different top_k values interactively in your app?
You said:
What are the trade-offs between different SentenceTransformer models?
ChatGPT said:

Fantastic question — the SentenceTransformer model you choose has a major impact on your RAG system’s:

    🔍 Retrieval accuracy (semantic matching)

    ⚡ Speed and memory usage

    📦 Disk space & download size

Here’s how to think about it.
🧠 What Varies Between SentenceTransformer Models?
Factor	Description
Architecture	MiniLM, DistilRoBERTa, MPNet, BERT, etc. — some are optimized for speed, others for depth.
Dimensionality	Ranges from 384 to 1024+. Affects FAISS index size and matching granularity.
Training data	Some trained on general web data, others on scientific/biomedical corpora.
Inference time	Smaller models run faster — especially important in real-time apps.
🔬 Comparison of Popular Models
Model Name	Dim	Speed (↑ = faster)	Quality (↑ = better)	Comments
all-MiniLM-L6-v2	384	🔹🔹🔹🔹🔹	🔹🔹🔹🔹	Small, fast, great for QA/semantic search
all-distilroberta-v1	768	🔹🔹🔹	🔹🔹🔹🔹🔹	Deeper, better nuance but slower
all-mpnet-base-v2	768	🔹🔹🔹	🔹🔹🔹🔹🔹🔹	One of the best general-purpose ST models
paraphrase-MiniLM-L6-v2	384	🔹🔹🔹🔹🔹	🔹🔹🔹	Good for paraphrase detection
multi-qa-MiniLM-L6-cos-v1	384	🔹🔹🔹🔹🔹	🔹🔹🔹🔹	Optimized for QA-style retrieval tasks
⚖️ Trade-Offs
Factor	Smaller Model (e.g., MiniLM)	Larger Model (e.g., MPNet, DistilRoBERTa)
🔄 Speed	Very fast — great for interactive use	Slower, but okay for batch or background tasks
🧠 Accuracy	Slightly lower nuance on complex queries	Better semantic depth & contextual precision
🧮 Memory use	Low — suitable for edge devices, Raspberry Pi	High — needs more RAM, GPU optional
🧊 Embedding size	Smaller FAISS index, faster search	Larger FAISS index, more RAM required
✅ Practical Advice

    Use all-MiniLM-L6-v2 if you want a fast, good-enough baseline.

    Use all-mpnet-base-v2 or all-distilroberta-v1 if you want higher-quality matching and don’t mind a performance hit.

    If you're building a QA-heavy app, test multi-qa-MiniLM-L6-cos-v1.

🧪 Want to Compare Models?

Try swapping out model_name in your RAG script:

model_name = "sentence-transformers/all-mpnet-base-v2"

Then time how long it takes to:

    Load the model

    Encode the chunks

    Answer 1–2 test questions

You can even add a benchmark mode to print time and top-5 retrieved chunks.

Want help setting up a benchmarking script to compare multiple models side-by-side?
You said:
How can we evaluate or measure the quality and accuracy of the RAG system’s answers without relying solely on human judgment?
ChatGPT said:

Excellent — this question gets into the heart of AI evaluation, which is tricky but essential. Let’s break down how to evaluate a RAG system’s output without relying purely on human opinion.
✅ 1. Use Reference-Based Metrics (if ground-truth exists)

If you have expected answers or a dataset of Q&A pairs, you can compare generated answers to those using standard NLP metrics:
Metric	What it Measures	When to Use
BLEU	N-gram overlap (good for precision)	Short factual answers
ROUGE	Recall of n-grams (good for coverage)	Longer summaries or responses
METEOR	Semantic match + synonym handling	Balanced QA tasks
BERTScore	Cosine similarity between BERT embeddings	Semantic similarity without exact match

    ✅ Tools like Evaluate from HuggingFace make this easy:

from evaluate import load
rouge = load("rouge")
results = rouge.compute(predictions=[output], references=[expected_answer])

✅ 2. Precision@k / Recall@k for Retrieval Evaluation

If you want to evaluate FAISS retrieval quality, especially without generation:

    Define a list of questions with known relevant source chunks.

    Use FAISS to get top-k results for each question.

    Compute:

        Precision@k = fraction of top-k that are correct

        Recall@k = fraction of all correct chunks found in top-k

This helps you test whether your retriever is surfacing the right context, regardless of generation.
✅ 3. Automated Factuality Check (Optional)

Use models like TruthfulQA or G-Eval to judge factual accuracy of generated answers.

    This is cutting-edge but powerful if you're scaling. Some tools even evaluate hallucinations directly.

✅ 4. Heuristics + Logging

Even without reference answers:

    Check: Does the answer contain keywords from the question?

    Log: Top retrieved chunks and final answer → spot bad retrieval or irrelevant generation.

    Score: Simple rules like “was the entity mentioned?” or “was the date correct?”

✅ 5. Manual Sampling Strategy

Even though we’re avoiding full human judgment:

    Sample 5–10 answers per model/config

    Use a rubric: Correct, Partially Correct, Incorrect, Irrelevant

    Use that to calibrate automated metrics and set thresholds

🧪 TL;DR: Combine Methods
Goal	Suggested Approach
Measure answer quality	ROUGE / METEOR / BERTScore
Measure retrieval accuracy	Precision@k / Recall@k
Detect hallucinations	Logging + TruthfulQA (optional)
Compare configs or models	Manual sampling + metric trends

Would you like help creating a small test set and benchmarking your RAG system with evaluate or rouge/bertscore?