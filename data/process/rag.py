from pathlib import Path


KNOWLEDGE_BASE_PATH = Path("docs/company-knowledge-base")


def load_documents():
    documents = []

    for file in KNOWLEDGE_BASE_PATH.glob("*.md"):
        text = file.read_text(encoding="utf-8")

        documents.append({
            "source_document": file.name,
            "text": text
        })

    return documents


def split_into_blocks(text):
    blocks = text.split("\n\n")
    return blocks


documents = load_documents()

loyalty_text = documents[0]["text"]
blocks = split_into_blocks(loyalty_text)

for index, block in enumerate(blocks):
    print(f"\n--- BLOQUE {index} ---")
    print(block)