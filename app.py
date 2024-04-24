# IMPORTING 
from util import generate_context,display_footer,write_answer
from text_to_speech import speak
from dataset.dataset import load_text_dataset
from indices.index import load_index
from encoder.encoder import load_encoder
from LLM.LLM import infer
import streamlit as st
import faiss
import tempfile
# from translator import translate
import httpcore
from dotenv import load_dotenv,find_dotenv,dotenv_values
from huggingface_hub import hf_hub_download
# from config import get_config
import os


print("HF_DATASET_CHECKPOINT",os.getenv('HF_DATASET_CHECKPOINT'))
print("FAISS_INDEX_FILE_PATH",os.getenv('FAISS_INDEX_FILE_PATH'))

setattr(httpcore, 'SyncHTTPTransport', None)

# SOME STATIC VARIABLES
k=10
max_line_length = 80
matching_threshold=0.50

language_choices = {
    'English': 'en',
    # 'Hindi': 'hi',
    # 'Gujarati': 'gu',
    # 'Marathi': 'mr',
    # 'Tamil': 'ta',
    # 'Telugu': 'te',
    # 'Kannada': 'kn',
    # 'Bengali': 'bn',
}

model_choices={
    "Mistral-7b-v0.2":"https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
    "gemma-7b":"https://api-inference.huggingface.co/models/google/gemma-7b"
}
# MAIN METHOD TO SET PAGE CONFIG
def main():
    
    st.set_page_config(
        page_title="Vivechan AI",
        page_icon="✨",
    )
    
if __name__ == "__main__":
    main()

# RETRIVING RESOURCES LIKE ENCODER , DATASET , INDEX etc.
@st.cache_resource
def get_cached_encoder():
    return load_encoder()

def get_dir_path():
    temp_dir = tempfile.gettempdir()
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir

@st.cache_resource
def load_cached_index(repo_id,repo_file_name):
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    temp_dir=get_dir_path()
    index_filepath=os.path.join(temp_dir,repo_file_name)
    if os.path.exists(index_filepath):
        print("index_file already exists ........ ")
        pass
    else:
        print("going to download index_file ...... ")
        hf_hub_download(
            repo_id=repo_id,
            filename=repo_file_name,
            local_dir = temp_dir,
            repo_type="dataset",
            token=HF_TOKEN
        )
        print("after download ..... ")
    
    print("Starting Loading Index .....",index_filepath)
    VectorIndex=faiss.read_index(index_filepath)
    print("Done Loading Index .....")
    return VectorIndex

@st.cache_resource
def get_cached_index():
    return load_index()

@st.cache_resource
def get_cached_text_dataset():
    return load_text_dataset()


Encoder=get_cached_encoder()

repo_id=os.getenv('VECTOR_STORE_REPO_ID')
repo_file_name=os.getenv('VECTOR_STORE_FILE_NAME')

print("repo_id : ",repo_id)
print("repo_file_name : ",repo_file_name)


VectorIndex=load_cached_index(repo_id,repo_file_name)
# VectorIndex=get_cached_index()

Texts=get_cached_text_dataset()


# UI
st.title("Vivechan AI 🌟")
st.subheader("AI for spritual matters")
st.markdown(
    """
    <style>
        .reportview-container {
            width: 90%;
        }
    </style>
    """,
    unsafe_allow_html=True
)
query = st.text_input("Ask any question related spritual matters i.e. Shiv Mahapuran, Shrimad Bhagwat , Shripad Charitramrutam : ")

# SELECTING LANGUAGE
language=language_choices[st.selectbox("Select Language:", list(language_choices.keys()))]
print("language : ",language)

llm_model=model_choices[st.selectbox("Select Model:", list(model_choices.keys()))]
print("llm_model : ",llm_model)

def ask(IsContinue=False):

    PreviousAnswer=st.session_state.get('PreviousAnswer','')
    Answer=st.session_state.get('Answer','')
    
    # Translating query
    translated_query=query
    # if language != 'en':
    #     translated_query=translate(query,language,'en')
    
    # encoding and retriving context form vector index
    encoded=Encoder.encode([translated_query])
    Distance,Positions=VectorIndex.search(encoded,k)
    Distance=Distance[0]
    Positions=Positions[0]
    min_distance=min(Distance)
    max_distance=max(Distance)
    if max_distance==min_distance:
        max_distance+=0.001

    Similarity=[(1-(dist-min_distance)/(max_distance-min_distance)) for dist in Distance]
    
    BetterPositions=[ Pos for i,Pos in enumerate(Positions) if Similarity[i] >= matching_threshold]

    print("Distance : ",Distance)
    print("Similarity : ",Similarity)

    print("Positions : ",Positions)
    print("BetterPositions : ",BetterPositions)
    print("Total Len of text : ",len(Texts))

    Context=generate_context(Texts,BetterPositions)

    
    
    # generating answer from the context
    CurrentAnswer=infer(translated_query,Context,llm_model)

    # Future feature : to support continue generation of previous answer
    if IsContinue : Answer+=CurrentAnswer
    else : Answer=CurrentAnswer
    PreviousAnswer=Answer
    
    # st.session_state['PreviousAnswer']=PreviousAnswer
    # st.session_state['Answer']=Answer

    write_answer(Answer,max_line_length,language)
    # st.session_state['ShouldContinue']=True

    # text to speech
    speak(Answer)

    st.subheader("Full Context : ")
    st.write(Context)

if st.button('Ask'):
    st.session_state['PreviousAnswer']=''
    st.session_state['Answer']=''
    st.session_state['ShouldContinue']=False
    ask()
    
display_footer()