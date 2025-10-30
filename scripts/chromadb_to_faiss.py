# migrate_chroma_to_faiss.py
import chromadb
import faiss
import numpy as np
import json
import os
from tqdm import tqdm

# === CONFIG ===
CHROMA_PATH = "chroma_db"        # Your Chroma persistent path
OUTPUT_DIR = "faiss"        # Where to save .bin + .json
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Optional: Use GPU (uncomment if you have CUDA)
# USE_GPU = True
# gpu_id = 0

# === Initialize Chroma ===
client = chromadb.PersistentClient(path=CHROMA_PATH)
collections = client.list_collections()

print(f"Found {len(collections)} collections. Starting migration...")

for coll in tqdm(collections, desc="Migrating collections"):
    name = coll.name
    collection = client.get_collection(name)

    # === Export in chunks (safe for large collections) ===
    limit = 10_000
    offset = 0
    all_vectors = []
    all_ids = []
    all_docs = []
    all_metas = []

    while True:
        results = collection.get(
            limit=limit,
            offset=offset,
            include=['embeddings', 'documents', 'metadatas']
        )
        if not results['ids']:
            break

        vectors = np.array(results['embeddings']).astype('float32')
        all_vectors.append(vectors)
        all_ids.extend(results['ids'])
        all_docs.extend(results['documents'])
        all_metas.extend(results['metadatas'])

        offset += limit

    if not all_vectors:
        print(f"  → {name}: empty, skipping")
        continue

    vectors = np.vstack(all_vectors)
    dim = vectors.shape[1]

    # === Build FAISS Index ===
    print(f"  → {name}: {len(vectors)} vectors, dim={dim}")

    # Choose index type: IVF for large, Flat for small
    if len(vectors) > 50_000:
        nlist = min(100, len(vectors) // 100)
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        index.train(vectors)
    else:
        index = faiss.IndexFlatL2(dim)  # Exact search for small data

    index.add(vectors)

    # Optional: Move to GPU
    # if USE_GPU:
    #     res = faiss.StandardGpuResources()
    #     index = faiss.index_cpu_to_gpu(res, gpu_id, index)

    # === Save FAISS Index ===
    index_path = os.path.join(OUTPUT_DIR, f"{name}.bin")
    faiss.write_index(index, index_path)

    # === Save Metadata ===
    meta_path = os.path.join(OUTPUT_DIR, f"{name}_meta.json")
    with open(meta_path, 'w') as f:
        json.dump({
            "ids": all_ids,
            "documents": all_docs,
            "metadatas": all_metas
        }, f)

    print(f"  → Saved: {index_path} + {meta_path.split('/')[-1]}")

print(f"\nMigration complete! All collections saved to: {OUTPUT_DIR}")