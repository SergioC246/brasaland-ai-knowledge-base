from pathlib import Path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


KNOWLEDGE_BASE_PATH = Path("docs/company-knowledge-base")
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

model = SentenceTransformer(EMBEDDING_MODEL)
client = QdrantClient(url="http://localhost:6333")
print(client.get_collections())


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

def chunk_document(document):
    blocks = split_into_blocks(document["text"])

    title = blocks[0].removeprefix("# ")
    content_blocks = blocks[1:]

    chunks = []

    for index, block in enumerate(content_blocks):
        chunk = {
            "company": "brasaland",
            "source_document": document["source_document"],
            "section": title,
            "language": "es",
            "chunk_index": index,
            "text": block
        }

        chunks.append(chunk)

    return chunks    


documents = load_documents()

all_chunks = []

for document in documents:
    document_chunks = chunk_document(document)
    all_chunks.extend(document_chunks)

embeddings = []

for chunk in all_chunks:
    embedding = model.encode(chunk["text"])
    embeddings.append(embedding)


print("Documentos:", len(documents))
print("Chunks totales:", len(all_chunks))
print("Embeddings generados:", len(embeddings))
print("Dimensiones del primer embedding:", len(embeddings[0]))