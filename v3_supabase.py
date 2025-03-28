from PIL import Image
import cohere
import base64
from supabase import create_client
import os
import numpy as np

# Initialize clients
co = cohere.ClientV2(api_key="your-cohere-api-key")
supabase = create_client(
    supabase_url="your-supabase-url",
    supabase_key="your-supabase-anon-key"
)

# First, create the table in Supabase SQL editor:
"""
-- Enable the vector extension
create extension if not exists vector;

-- Create a table to store image embeddings
create table if not exists image_embeddings (
  id bigint primary key,
  file_path text,
  file_name text,
  embedding vector(1024),  -- Cohere's embed-multilingual-v3.0 dimension
  created_at timestamp with time zone default timezone('utc'::text, now())
);

-- Create a vector index
create index on image_embeddings 
using ivfflat (embedding vector_cosine_ops)
with (lists = 100);
"""

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
    """Load images from folder and create embeddings in Supabase."""
    files = os.listdir(folder_path)
    
    for i, file in enumerate(files):
        file_path = os.path.join(folder_path, file)
        
        # Get embedding
        res = image_to_base64_data_url(file_path)
        embedding = res.embeddings.float[0]
        
        # Insert into Supabase
        data = {
            'id': i,
            'file_path': file_path,
            'file_name': file,
            'embedding': embedding
        }
        
        supabase.table('image_embeddings').insert(data).execute()
    
    return len(files)

def retrieve_images(query, top_k=5):
    """Retrieve similar images based on text query."""
    # Convert query to embedding
    query_emb = co.embed(
        texts=[query],
        model="embed-multilingual-v3.0",
        input_type="search_query",
        embedding_types=["float"]
    ).embeddings.float[0]
    
    # Search in Supabase using vector similarity
    rpc_response = supabase.rpc(
        'match_images',  # We'll create this function below
        {
            'query_embedding': query_emb,
            'match_count': top_k
        }
    ).execute()
    
    return rpc_response.data

def display_results(results, size=(200, 200)):
    """Display search results with scores."""
    print("-" * 100)
    print("Top matches:")
    for i, result in enumerate(results):
        print(f"Ranking {i+1} with similarity score: {result['similarity']:.2f}")
        print(f"File: {result['file_name']}")
        
        # Open and resize image
        img = Image.open(result['file_path'])
        img_resized = img.resize(size)
        display(img_resized)  # This works in Jupyter notebooks
        print("-" * 50)

# Create the matching function in Supabase SQL editor:
"""
create or replace function match_images (
  query_embedding vector(1024),
  match_count int
)
returns table (
  id bigint,
  file_path text,
  file_name text,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    id,
    file_path,
    file_name,
    1 - (embedding <=> query_embedding) as similarity
  from image_embeddings
  order by embedding <=> query_embedding
  limit match_count;
end;
$$;
"""

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
