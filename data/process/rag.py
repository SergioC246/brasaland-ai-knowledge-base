# -------------------------------------------------
# IMPORTS
# Los imports nos permiten utilizar código creado en otras librerías o módulos dentro de nuestro programa
# -------------------------------------------------

import uuid
# uuid nos permite generar identificadores únicos. Los usamos para que cada punto guardado en Qdrant tenga un ID.
# En nuestro caso usamos UUID5 porque genera IDs deterministas: si le damos la misma informaci´ón, genera siempre el mismo ID.

from pathlib import Path
# Path nos ayuda a trabajar con rutas de archivos y carpetas de una forma más cómoda que utilizando strings normales

from sentence_transformers import SentenceTransformer
# SentenceTransformer nos permite cargar nuestros modelos de embeddings.
# El modelo convierte texto en vectores númericos que representan aproximadamente su significado semántico

from qdrant_client import QdrantClient
# QdranClient es el cliente de Pythin que nos permite comunicarnos con nuestro servicor de Qdrant

from qdrant_client.models import Distance, VectorParams, PointStruct
# Distance: permite elegir cómo compararemos los vectores.
# VectorParams: permite configurar los vectores de nuestra colección, por ejemplo su tamaño y la métrica de distancia
# PointStruc: representa un punto que guardamos en Qdrant. Cada punto contiene pricipalmente: ID + VECTOR + PAYLOAD

# -------------------------------------------------
# CONSTANTES / CONFIGURACIÓN
# Estas variables contienen valores de configuración que utilizzazremos en diferentes partes del programa.
# Las escribimos en MAYÚSCULAS por convención para indicar que no esperamos modificales durante la ejecución.
# -------------------------------------------------

KNOWLEDGE_BASE_PATH = Path("docs/company-knowledge-base")
# Ruta de la carpeta donde están los documentos .md que forman la base de conocimientos de Brasaland

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
# Nombre del modelo de embeddings que utilizamos. Su trabajo es convertir texto en vectores numéricos para poder comparar significado semántico

COLLECTION_NAME = "brasaland_knowledge"
# Nombre de la colección de Qdrant donde guardamos los puntos de nuestra base de conocimiento

VECTOR_SIZE = 384
# Nuestro modelo genera un vector de 384 números por cada texto.
# Por eso Qdrant debe crear la colección esperando vectores de exactamente 384 dimensiones

MIN_SCORE = 0.85
# Umbral mínimo de simulitud que estamos probando. Un resultado con score inferior a 0.85 se descarta

# -------------------------------------------------
# MODELO DE EMBEDDINGS Y CONEXIÓN CON QDRANT
# -------------------------------------------------

model = SentenceTransformer(EMBEDDING_MODEL)
# Carga el modelo de embeddings en memoria. Después podremos utilizar: model.encode(texto) para convetir texto en un vector
 
client = QdrantClient(url="http://localhost:6333")
# Crea el cliente que utilizará Pythin para comunicarse con Qdrant.
# Qdrant está ejecutandose en Docker y exponemos su puerto HTTP 6333.

# -------------------------------------------------
# SETUP DE QDRANT
# -------------------------------------------------

def setup():
    # def crea/define una función. Una función agrupa instrucciones que podemos ejecutar posteriomente llamando a su nombre
    collection_exists = client.collection_exists(
        collection_name=COLLECTION_NAME
    )
    # Preguntamos a Qdrant si nuestra colección ya existe. El resultado será un booleano: True: existe o False: no existe

    if not collection_exists:
        # if permite ejecutar código solamente cuando se cumple una condición
        # "not" invierte el boolenao.
        # Por tanto: if not collection_exists significa: "SI la colección no existe..."
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                # Qdrant espera vectores de 384 dimensiones
                distance=Distance.COSINE
                # Utilizamos similitud/distancia coseno para comparar que tan parecida es la direcciín de dos vectores.
                # Esto nos ayuda a encontrar textos semánticos similares
            )
        )
    
    print("¿Existe la colección?", collection_exists)
    
setup()
# Aquí LLAMAMOS a la función. 
# def setup() -> DEFINE la función. 
# setup() -> EJECUTA la función

# -------------------------------------------------
# SETUP DE QDRANT
# -------------------------------------------------

def generate_point_id(chunk):
    # Esta función RECIBE un chunk mediante el parámetro "chunk"

    unique_text = f"{chunk['source_document']}:{chunk['chunk_index']}"
    # Creamos un string utilizando:
    # - nombre del documento
    # - posición del chunk
    # Ejemplo: brasaland-waste-protocol.es.md:2
    # f".." es una f-string. Nos permite insetar valores dentro de un string utilizando {}

    return str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_text))
    # uuid5 genera siempre el mismo UUID cuando recibe el mismo unique_text
    # Esto hace nuestros IDs DETERMINISTAS.
    # Si ejecutamos la indexación otra vez, el mismo chunk tendrá el mismo ID y Qdrant podrá actulizarlo en lugar de
    # crear necesariemente otro punto diferente

# -------------------------------------------------
# SETUP DE QDRANT
# -------------------------------------------------

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


points = []

for index, chunk in enumerate(all_chunks):
    point = PointStruct(
        id=generate_point_id(chunk),
        vector=embeddings[index].tolist(),
        payload=chunk
    )

    points.append(point)

client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

count_result = client.count(
    collection_name=COLLECTION_NAME,
    exact=True
)

question = "Qué debo hacer si se desperdician más de 2 kg de carne en un turno?"

question_embedding = model.encode(question)

search_result = client.query_points(
    collection_name=COLLECTION_NAME,
    query=question_embedding.tolist(),
    limit=3,
    with_payload=True    
)

relevant_results = []
for result in search_result.points:
    if result.score >= MIN_SCORE:
        relevant_results.append(result)
        #print("Score:", result.score)
        #print("Texto:", result.payload["text"])
        #print("---")#

print("Resultados relevantes:", len(relevant_results))
print("Dimensiones pregunta:", len(question_embedding))
print("Points almacenados en Qdrant:", count_result.count)
print("Points creados:", len(points))
print("Documentos:", len(documents))
print("Chunks totales:", len(all_chunks))
print("Embeddings generados:", len(embeddings))
print("Dimensiones del primer embedding:", len(embeddings[0]))