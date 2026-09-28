import uuid
from pathlib import Path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


KNOWLEDGE_BASE_PATH = Path("docs/company-knowledge-base")
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
COLLECTION_NAME = "brasaland_knowledge"
VECTOR_SIZE = 384

model = SentenceTransformer(EMBEDDING_MODEL)
client = QdrantClient(url="http://localhost:6333")
def setup():
    collection_exists = client.collection_exists(
        collection_name=COLLECTION_NAME
    )

    if not collection_exists:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )
    
    print("¿Existe la colección?", collection_exists)

setup()    
print(client.get_collections())

def generate_point_id(chunk):
    unique_text = f"{chunk['source_document']}:{chunk['chunk_index']}"

    return str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_text))


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

print("ID primer chunk:", generate_point_id(all_chunks[0]))
print("ID primer chunk otra vez", generate_point_id(all_chunks[0]))    

embeddings = []

for chunk in all_chunks:
    embedding = model.encode(chunk["text"])
    embeddings.append(embedding)


print("Documentos:", len(documents))
print("Chunks totales:", len(all_chunks))
print("Embeddings generados:", len(embeddings))
print("Dimensiones del primer embedding:", len(embeddings[0]))