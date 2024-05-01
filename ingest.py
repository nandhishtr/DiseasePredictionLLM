import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant

embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

print(embeddings)

loader = DirectoryLoader('./', glob="health_report_*.txt", show_progress=True, loader_cls=UnstructuredFileLoader)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)

texts = text_splitter.split_documents(documents)

print(texts[1])

# speify url of Qdrant
url = "http://localhost:6333"
qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    url=url,
    prefer_grpc=False,
    collection_name="vector_db"
)

print("Vector DB Successfully Created!")



"""
def create_vector_db(model_name, data_folder_path, url):
    embeddings = SentenceTransformerEmbeddings(model_name=model_name)
    loader = DirectoryLoader(data_folder_path, glob="health_report_*.txt", show_progress=True,
                             loader_cls=UnstructuredFileLoader)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)
    texts = text_splitter.split_documents(documents)


    # Create Qdrant index from the documents
    qdrant = Qdrant.from_documents(
        texts,
        embeddings,
        url=url,
        prefer_grpc=False,
        collection_name="vector_db"
    )

    print("Vector DB Successfully Created!")


# Define parameters
if __name__ == "__main__":
    model_name = "NeuML/pubmedbert-base-embeddings"
    data_folder_path = "/Users/asminanasser/PycharmProjects/pythonProject3/dataset_folder/health_report_*"
    url = "http://localhost:6333"

    # Call the method
    create_vector_db(model_name, data_folder_path, url)


"""

