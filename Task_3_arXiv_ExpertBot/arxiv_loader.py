import pandas as pd

def load_arxiv_subset(json_path="arxiv-metadata-oai-snapshot.json", category="cs", limit=500):
    """
    Load a subset of the arXiv dataset based on the category prefix (e.g., 'cs').
    
    Args:
        json_path (str): Path to the arXiv metadata JSON file.
        category (str): arXiv category prefix (e.g., 'cs', 'math').
        limit (int): Max number of papers to load.

    Returns:
        List[Dict]: Filtered papers with 'content' field containing title + abstract.
    """
    df = pd.read_json(json_path, lines=True)

    # Filter rows where 'categories' contains the given prefix (e.g., "cs.")
    df = df[df["categories"].str.contains(f"^{category}\\.", regex=True)]

    # Remove rows with missing abstracts or titles
    df = df.dropna(subset=["title", "abstract"])

    # Limit the number of records
    df = df.head(limit)

    # Create content blocks
    docs = []
    for _, row in df.iterrows():
        content = f"Title: {row['title'].strip()}\n\nAbstract: {row['abstract'].strip()}"
        docs.append({"content": content})

    return docs

if __name__ == "__main__":
    data = load_arxiv_subset(category="cs", limit=500)
    print(f"✅ Loaded {len(data)} documents from arXiv dataset.")
