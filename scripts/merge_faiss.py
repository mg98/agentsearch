import faiss
import json
from pathlib import Path
from tqdm import tqdm

def merge_faiss_indices(faiss_dir: str, output_name: str = "all_agents"):
    faiss_path = Path(faiss_dir)

    bin_files = sorted(faiss_path.glob("agent_*_meta.json"))
    agent_ids = [int(f.stem.split("_")[1]) for f in bin_files]

    print(f"Found {len(agent_ids)} agent indices")

    first_index = faiss.read_index(str(faiss_path / f"agent_{agent_ids[0]}.bin"))
    dimension = first_index.d

    merged_index = faiss.IndexFlatL2(dimension)

    all_ids = []
    all_documents = []
    all_metadatas = []

    for agent_id in tqdm(agent_ids, desc="Merging indices"):
        index_path = faiss_path / f"agent_{agent_id}.bin"
        meta_path = faiss_path / f"agent_{agent_id}_meta.json"

        index = faiss.read_index(str(index_path))
        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        vectors = faiss.rev_swig_ptr(index.get_xb(), index.ntotal * dimension).reshape(index.ntotal, dimension)
        merged_index.add(vectors)

        all_ids.extend(metadata['ids'])
        all_documents.extend(metadata['documents'])

        agent_metadatas = [{'agent_id': agent_id} if meta is None else {**meta, 'agent_id': agent_id}
                          for meta in metadata['metadatas']]
        all_metadatas.extend(agent_metadatas)

    output_index_path = faiss_path / f"{output_name}.bin"
    output_meta_path = faiss_path / f"{output_name}_meta.json"

    faiss.write_index(merged_index, str(output_index_path))

    merged_metadata = {
        'ids': all_ids,
        'documents': all_documents,
        'metadatas': all_metadatas
    }

    with open(output_meta_path, 'w') as f:
        json.dump(merged_metadata, f)

    print(f"\nMerged index saved to {output_index_path}")
    print(f"Total vectors: {merged_index.ntotal}")
    print(f"Total documents: {len(all_documents)}")
    print(f"Metadata saved to {output_meta_path}")

if __name__ == "__main__":
    merge_faiss_indices("faiss")
