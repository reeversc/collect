from PIL import Image
import cohere
import base64
from qdrant_client import QdrantClient
from qdrant_client.http import models
import os

# Initialize clients
co = cohere.ClientV2(api_key="your-cohere-api-key")
qdrant_client = QdrantClient(
    url="your-qdrant-url",
    api_key="your-qdrant-api-key"
)

# Collection configuration
collection_name = "image_search"
vector_size = 1024  # Cohere's embed-multilingual-v3.0 dimension

# Create collection
qdrant_client.recreate_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=vector_size,
        distance=models.Distance.COSINE
    )
)

def image_to_base64_data_url(image_path):
    """Convert image to base64 and get embedding from Cohere."""
    with open(image_path, "rb") as f:
        enc_img = base64.b64encode(f.read()).decode("utf-8")
        enc_img = f"data:image/jpeg;base64,{enc_img}"

    response = co.embed(
        model="embed-multilingual-v3.0",
        images=[enc_img],
        input_type="image",
        embedding_types=["float"],
    )

    return response

def load_and_embed_images(folder_path):
    """Load images from folder and create embeddings."""
    files = os.listdir(folder_path)
    points = []
    
    for i, file in enumerate(files):
        file_path = os.path.join(folder_path, file)
        
        # Get embedding
        res = image_to_base64_data_url(file_path)
        embedding = res.embeddings.float[0]
        
        # Create point
        point = models.PointStruct(
            id=i,
            vector=embedding,
            payload={
                "file_path": file_path,
                "file_name": file
            }
        )
        points.append(point)
    
    # Upload points to collection
    qdrant_client.upsert(
        collection_name=collection_name,
        points=points
    )
    
    return len(points)

def retrieve_images(query, top_k=5):
    """Retrieve similar images based on text query."""
    # Convert query to embedding
    query_emb = co.embed(
        texts=[query],
        model="embed-multilingual-v3.0",
        input_type="search_query",
        embedding_types=["float"]
    ).embeddings.float[0]
    
    # Search in Qdrant
    search_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_emb,
        limit=top_k
    )
    
    results = []
    for res in search_results:
        results.append({
            'file_path': res.payload['file_path'],
            'score': res.score,
            'file_name': res.payload['file_name']
        })
    
    return results

def display_results(results, size=(200, 200)):
    """Display search results with scores."""
    print("-" * 100)
    print("Top matches:")
    for i, result in enumerate(results):
        print(f"Ranking {i+1} with similarity score: {result['score']:.2f}")
        print(f"File: {result['file_name']}")
        
        # Open and resize image
        img = Image.open(result['file_path'])
        img_resized = img.resize(size)
        display(img_resized)  # This works in Jupyter notebooks
        print("-" * 50)

# Example usage:
if __name__ == "__main__":
    # 1. Load and index images
    folder_path = "data/multimodal_semantic_search"
    num_indexed = load_and_embed_images(folder_path)
    print(f"Indexed {num_indexed} images")
    
    # 2. Perform searches
    queries = [
        "People wearing jewelry",
        "People wearing green",
        "People wearing glasses"
    ]
    
    for query in queries:
        print(f"\nSearching for: {query}")
        results = retrieve_images(query, top_k=2)
        display_results(results)
