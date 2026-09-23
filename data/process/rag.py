from pathlib import Path
from sentence_transformers import SentenceTransformer


KNOWLEDGE_BASE_PATH = Path("docs/company-knowledge-base")
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

model = SentenceTransformer(EMBEDDING_MODEL)


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


print("Documentos:", len(documents))
print("Chunks totales:", len(all_chunks))

first_chunk = all_chunks[0]

text = first_chunk["text"]

embedding = model.encode(text)

print("Texto:")
print(text)

print("Tipo:", type(embedding))
print("Dimensiones:", len(embedding))
print("Primeros 5 valores:", embedding[:5])