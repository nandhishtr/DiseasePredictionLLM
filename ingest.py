from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import CTransformers

#initializes an embedding model
embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

# load the current health report documents
loader = DirectoryLoader("health_reports", glob="**/*.txt", show_progress=True)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)

pages = loader.load_and_split(text_splitter)

#vector store
db = Chroma.from_documents(pages, embeddings, persist_directory='./chroma_db')

#llm initialization
llm = CTransformers(model="MaziyarPanahi/BioMistral-7B-GGUF", model_file="BioMistral-7B.Q4_K_M.gguf", temperature=0.7,
                    max_tokens=2048, top_p=1, n_ctx=2048, config={'context_length': 2048})
