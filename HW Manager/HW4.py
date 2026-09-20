import streamlit as st
from openai import OpenAI
import sys
from bs4 import BeautifulSoup
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import chromadb
from pathlib import Path

#Page Title
st.title("HW 4- iSchool Student Organization Chatbot")

#Open AI Client
if "openai_client" not in st.session_state:
    st.session_state.openai_client = OpenAI(
        api_key=st.secrets["OPEN_API_KEY"]
    )


#To extract text from a PDF file
def extract_text_from_html(html_path):
    with open(html_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file,"html.parser")

        text = soup.get_text(
            separator=" ",
            strip= True
        
        )

    return text
#Split html into 2 chunks
#Keeps the chunks similar in size 
#Allow both parts of the doc to be stored 
#and searched seperately in the vector database
def split_text_two_chunks(text):
    middle = len(text) // 2
    chunk1 = text[:middle]
    chunk2 = text[middle:]

    return chunk1, chunk2
#Add text to the ChromaDB collection
def add_to_collection(collection, text, file_name):

    client = st.session_state.openai_client

    #Create embedding
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )

    embedding = response.data[0].embedding

    #Store document in ChromaDB
    collection.add(
        ids=[file_name],
        documents=[text],
        metadatas=[
            {"file_name": file_name}
        ],
        embeddings=[embedding]
    )
#Load PDFs from a folder into the ChromaDB collection
def load_html_to_collection(folder_path, collection):

    html_folder = Path(folder_path)

    html_files = list(
        html_folder.glob("*.html")
    )

    for html_file in html_files:
    #Extract text from HTML  
        
        text = extract_text_from_html(
            html_file
        )

        chunk1, chunk2 = split_text_two_chunks(
            text
        )

        file_name = html_file.name
#Store Chunk 1 
        add_to_collection(
            collection,
            chunk1,
            file_name + "_chunk1"
        )
#Store Chunk 2
        add_to_collection(
            collection,
            chunk2,
            file_name + "_chunk2"

        )
    return len(html_files)
#Create vector database 
def create_vector_db():

    #create/open ChromaDB client
    chroma_client = chromadb.PersistentClient(
        path="./ChromaDB_for_HW4"
    )

    collection = chroma_client.get_or_create_collection(
        name="HW4Collection"
    )

    #Only create embeddings if the collection is empty
    if collection.count() == 0:

        load_html_to_collection(
            "HW Manager/HW4-Data/su_orgs",
            collection
        )

    return collection


#Store vector

if "HW4_VectorDB" not in st.session_state:

    st.session_state.HW4_VectorDB = (
        create_vector_db()
    )

#To search the vector database
def search_vector_db(collection, search_text):

    client = st.session_state.openai_client

    # Convert the user's search into an embedding
    response = client.embeddings.create(
        input=search_text,
        model="text-embedding-3-small"
    )

    query_embedding = (
        response.data[0].embedding
    )

    # Find the 3 closest documents
    results = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=3
    )

    return results

#Part B: Course chatbot display
if "HW4_messages" not in st.session_state:

    st.session_state.HW4_messages = []


#Display previous messages
for message in st.session_state.HW4_messages:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )


#Get question from user
prompt = st.chat_input(
    "Ask a question about student organizations..."
)


if prompt:

    #Display/save user message
    with st.chat_message("user"):
        st.write(prompt)

    st.session_state.HW4_messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

#Info from the vector database
    results = search_vector_db(
        st.session_state.HW4_VectorDB,
        prompt
    )

    retrieved_documents = (
        results["documents"][0]
    )

    retrieved_files = (
        results["ids"][0]
    )


    #Combine retrieved documents
    rag_context = ""

    for i in range(
        len(retrieved_documents)
    ):

        rag_context += (
            f"\n\nSource: "
            f"{retrieved_files[i]}\n"
            f"{retrieved_documents[i]}"
        )

#System prompt for the RAG pipeline
    system_prompt = f"""
You are a helpful Syracuse University student organization assistant.

Use the retrieved student organization information below to answer 
the user's question. 

If the retrieved information contains the answer,
use that information to answer the question. 

If the retrieved information does not contain the answer,
clearly say that the anser was not found in the
retrieved student organization information. 

Retrieved student organization information: 

{rag_context}
"""


    #Create messages sent to OpenAI
    messages_for_llm = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

#Keep up to the last 5 interactions
# Give only the last 10 messages from the list 
    recent_messages = st.session_state.HW4_messages[-10:]

    messages_for_llm.extend(
        recent_messages
    )
#Send messages to OpenAI and get response
    client = st.session_state.openai_client

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=messages_for_llm
    )

    assistant_response = (
        response.choices[0].message.content
    )


    #Display assistant response
    with st.chat_message("assistant"):

        st.write(
            assistant_response
        )

        st.caption(
            "RAG sources: "
            + ", ".join(retrieved_files)
        )


    #Save assistant response
    st.session_state.HW4_messages.append(
        {
            "role": "assistant",
            "content": assistant_response
        }
    )