from app.services.embeddings import generate_embedding

vec = generate_embedding("Hello world, this is a free model!")
print(f"Vector dimension: {len(vec)}")   # Should print 384
print(f"First 5 values: {vec[:5]}")       # Should show small floats