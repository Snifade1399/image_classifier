import json

with open("friends_embeddings.json", "r") as f:
    data = json.load(f)
    for name, embedding in data.items():
        print(f"{name} -> Length: {len(embedding)}")
        break  # remove break if you want to see all

