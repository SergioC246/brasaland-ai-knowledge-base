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
# GENERACION DE IDs
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
# CARGA DE DOCUMENTOS
# -------------------------------------------------

def load_documents():
    # Esta función leerá nuestros archivos .md y devolverá una lista con los documentos.

    documents = []
    # [] crea una lista Vacía. Aquí iremos guardando los documentos encontrados.

    for file in KNOWLEDGE_BASE_PATH.glob("*.md"):
        # for crea un BUCLE. Un bucle permite repetir instrucciones.

        # En este caso recorremos UNO POR UNO todos los archivos .md encontrados en KNOWLEDGE_BASE_PATH
        # En cada vuelta del bucle, "file" represnta el archivo que estamos procesando actualmente

        text = file.read_text(encoding="utf-8")
        # Lee el contenido del archivo y lo guarda en "text"
        # UTF-8 permite interpretar correctamente caracteres como á, é, ñ, etc.

        documents.append({
            "source_document": file.name,
            "text": text
        })
        # append añade UN elemento al final de una lista
        # En este caso añadimos un DICCIONARIO a documents. Un diccionario guarda información mediante: calve -> valor
        # Por ejemplo: "source_document" -> nombre del archivo. "text" -> contenido del archivo

    return documents
    # DEvuelve la lista completa cuando el for ha terminado

# -------------------------------------------------
# DIVISIÓN DEL TEXTO
# -------------------------------------------------

def split_into_blocks(text):
    # Recibe un texto completo.

    blocks = text.split("\n\n")
    # split divide un string utilizando un separador
    # "\n" significa salto de linea. "\n\n" significa dos saltos de linea.
    # Así dividimos nuestro documentos por bloques/párrafos.

    return blocks
    # Devuelve la lista de bloques resultantes.

# -------------------------------------------------
# CREACIÓN DE CHUNKS
# -------------------------------------------------

def chunk_document(document):
    # Recibe UN documento y lo transforma en varios chunks.

    blocks = split_into_blocks(document["text"])
    # Accedemos al valor asociado a la calve "text" del diccionario document.
    # Después lo dividimos en bloques

    title = blocks[0].removeprefix("# ")
    # Las listas empiezan en posición 0
    # blocks[0] = primer bloque del documento
    # En nuestros documentos ese primer bloque es el título.
    # removeprefix("# ") elimina el símbolo Markdown "# ".

    content_blocks = blocks[1:]
    # [1:] significa: "dame todos los elementos desde la posición 1 hasta el final".
    # Así dejamos fuera el título y conservamos los bloques que contienen información

    chunks = []
    # Lista vacía donde guardaremos los chunks de este documento.

    for index, block in enumerate(content_blocks):
        # for vuelve a recorrer elementos uno por uno
        # enumerate nos proporciona DOS valores:
        # index -> posición: 0, 1, 2, 3...
        # block -> contenido del bloque actual

        chunk = {
            "company": "brasaland",
            "source_document": document["source_document"],
            "section": title,
            "language": "es",
            "chunk_index": index,
            "text": block
        }
        #Creamos un diccionario que representa UN chunk.
        # Además del texto guardamos METADATOS
        # Estos metadatos formarán después el payload de Qdrant y nos permitirán saber de dónde salió la información

        chunks.append(chunk)
        # Añadimos el chunk actual a la lista de chunks

    return chunks   
    # Después de terminar el bucle devovemos todos los chunks creados para ese documento

# -------------------------------------------------
# CARGAMOS TODOS LOS DOCUMENTOS
# -------------------------------------------------

documents = load_documents()
# Ejecutamos load_documents().
# El resultado que devuelve mediante "return" se guarda en la variable documents.

# -------------------------------------------------
# CREAMOS LOS CHUNKS DE TODOS LOS DOCUMENTOS
# -------------------------------------------------

all_chunks = []
# Lista va´cia donde terminaremos teniendo los chunks de TODOS los documentos.

for document in documents:
    # Recorremos los documentos uno por uno.
    # En cada vuelta "document" contiene un documento.

    document_chunks = chunk_document(document)
    # Convertimos el documento actual en chunks.
    # El resultado es una LISTA de chunks.

    all_chunks.extend(document_chunks)
    # extend añade a una lista TODOS los elemtnos de otra lista.
    # Diferencia importante:
    # append(document_chunks): añadiría la lista completa como UN elemento.
    # extend(document_chunks): añade cada chunk individualmente.
    # QUeremos una única lista con todos los chunks.

# -------------------------------------------------
# GENERAMOS LOS EMBEDDINGS
# -------------------------------------------------

embeddings = []
# Lista vacía donde guardaremos los embeddings.

for chunk in all_chunks:
    # Recorremos todos los chunks uno por uno

    embedding = model.encode(chunk["text"])
    # Tomamos el texto del chunk actual y lo pasamos por nuestro modelo de embeddings.
    # El resultado es un VECTOR de 384 números que representa características semánticas del texto
    # IMPORTANTE: No tenemos 384 embedding. Tenemos UN embedding que contiene 384 dimesiones/números

    embeddings.append(embedding)
    # Guardamos el embedding actual en nuestra lista.
    # Tenemos 18 chunks -> tendremos 18 embeddings. Cada embedding -> 384 dimensiones

# -------------------------------------------------
# CONSTRUIMOS LOS PUNTOS PARA QDRANT
# -------------------------------------------------

points = []
# Lista donde guardaremos los puntos antes de enviarlos a Qdrant.

for index, chunk in enumerate(all_chunks):
    # Recorremos todos los chunks.
    # index nos permite saber qué posición estamos procesando. Lo utilizamos también para obtener el embedding correspondiente

    point = PointStruct(
        id=generate_point_id(chunk),
        # ID único y determinista del punto

        vector=embeddings[index].tolist(),
        # Obtenemos el embedding correspondiente al chunk
        # .tolist() convierte el array NumPY del embedding en una lista normal de Python
        
        payload=chunk
        # Payload = información que guardamos junto al vector.
        # Aquí contiene:
        # company
        # source_document
        # section
        # lenguage
        # chunk_index
        # text
        # Qdrant utiliza principalmente el VECTOR para buscar similtudes y nosotros recuperamos el PAYLOAD para obtener
        # texto y sus metadotas
    )

    points.append(point)
    # Añadimos el PointStruct recién creado a la lista points.

# -------------------------------------------------
# GUARDAMOS LOS PUNTOS EN QDRANT
# -------------------------------------------------

client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)
# upsert combina dos comportamientos:
# INSERT -> si el ID no existe, crea el punto.
# UPDATE -> si el ID ya existe, actualiza/reemplaza ese punto.
# Como unsamos IDs deterministas, ejecutar nuevamente el programa no debería crear duplicados de los mismos chunks.

# -------------------------------------------------
# CONTAMOS LOS PUNTOS ALMACENADOS
# -------------------------------------------------

count_result = client.count(
    collection_name=COLLECTION_NAME,
    exact=True
)
# Le preguntamos a Qdrant cuántos puntos existen realmente dentro de nuestra colección.

# -------------------------------------------------
# RETRIEVAL - BÚSQUEDA DE INFORMACIÓN
# -------------------------------------------------

question = "Qué debo hacer si se desperdician más de 2 kg de carne en un turno?"
# Esta es nuestra pregunta prueba
# Más adelante NO estará escrita directamente aquí: llegará desde retrieve(question) y posteriormente desde FastAPI.

question_embedding = model.encode(question)
# Convertimos la pregunta a un embedding.
# Es importante que pregunta y chunks puedan compararse dentro del mismo espacio vectorial

search_result = client.query_points(
    collection_name=COLLECTION_NAME,
    # Indicamos en qué colección queremos buscar.

    query=question_embedding.tolist(),
    # Enviamos el VECTOR de la pregunta a Qdrant.
    # tolist() convierte el array NumPy a una lista de Python
    # No enviamos simplemente la pregunta "question" porque la búsqueda semántica compara vectores.

    limit=3,
    # TOP_K = 3
    # Pedimos como máximo los 3 candidatos con mayor similitud
    # IMPORTANTE: Ser "uno de los 3 mejores" NO significa necesariamente que el resultado sea suficiemente relevante.

    with_payload=True
    # PEdimos a Qdrant que además del score/vector nos devuelva el payload con el texto y los metadatos del chunk    
)

# -------------------------------------------------
# FILTRAMOS LOS RESULTADOS POR SIMILITUD
# -------------------------------------------------

relevant_results = []
# Lista vacía donde guardamos únicamente los resultados que superen nuestro MIN_SCORE

for result in search_result.points:
    # Recorremos UNO POR UNO los candidatos devueltos por Qdrant.
    # "result" representa el candidato actual.

    if result.score >= MIN_SCORE:
        # Comprobamos si el score del resultado alcanza nuestra nota mínima.

        relevant_results.append(result)
        # Si pasa el filtro, guardamos el resultado en relevant_results.
        # Si NO pasa el if, Python simplemente no ejecuta esta línea para ese resultado


def retrieve(question):
# def define una función llamada retrieve.
# "question" es el parámetro que recibirá la pregunta.

    question_embedding = model.encode(question)
    # Convertimos la pregunta en un embedding de 384 dimensiones.

    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=question_embedding.tolist(),
        limit=3,
        with_payload=True
    )

    relevant_results = []
    # Creamos una lista vacía donde guardaremos solamente los resultados que consideremos relevantes.

    for result in search_result.points:
    # for recorre UNO POR UNO los resultados encontrados por Qdrant.
    # En cada vuelta, "result" representa un resultado diferente
        
        if result.score >= MIN_SCORE:
        # if comprueba una condición.
        # Solo entra aquí si el score del resultado es igual o superior a nuestra nota mínima.    
            relevant_results.append(result)
            # append añade UN elemento a la lista.
            # Si el resultado supera el filtro, lo guardamos dentro de releevant_results.

    return relevant_results
    # return devuelve fuera de la función los resultados que han superado el filtro MIN_SCORE              

print("Resultados relevantes:", len(relevant_results))
print("Dimensiones pregunta:", len(question_embedding))
print("Points almacenados en Qdrant:", count_result.count)
print("Points creados:", len(points))
print("Documentos:", len(documents))
print("Chunks totales:", len(all_chunks))
print("Embeddings generados:", len(embeddings))
print("Dimensiones del primer embedding:", len(embeddings[0]))