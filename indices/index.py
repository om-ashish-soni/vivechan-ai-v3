import faiss
import os
from dotenv import load_dotenv
# from config import get_config
import tempfile

def load_index(index_file_path):
    load_dotenv()
    # FAISS_INDEX_FILE_PATH=indices/vivechan-spiritual-v3.faiss
    # index_file_path=os.getenv('FAISS_INDEX_FILE_PATH')
    print("Starting Loading Index .....",index_file_path)
    VectorIndex=faiss.read_index(index_file_path)
    print("Done Loading Index .....")
    return VectorIndex



