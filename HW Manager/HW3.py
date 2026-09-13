import streamlit as st
from openai import OpenAI
from bs4 import BeautifulSoup
import requests
import google.generativeai as genai

#Page title

st.title("HW3 Streaming Chatbot that Discusses a URL")

st.write(
    "This chatbot uses content from one or two webpages to answer questions. "
    "The chatbot uses a 6-message conversation buffer, which keeps the "
    "three most recent user-assistant exchanges as short-term memory."
)
#To read url content
#From HW2
def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        return soup.get_text()

    except requests.RequestException as e:
        print(f"Error reading {url}: {e}")
        return None


#For sidebar/ url inputs
url1 = st.sidebar.text_input(
    "Enter first URL:"
)

url2 = st.sidebar.text_input(
    "Enter second URL (optional):"
)
#LLM selection

llm_choice = st.sidebar.selectbox(
    "Choose an LLM:",
    ["OpenAI", "Gemini"]
)
#API KEYS / CLIENTS

if "openai_client" not in st.session_state:
    openai_api_key = st.secrets["OPEN_API_KEY"]

    st.session_state.openai_client = OpenAI(
        api_key=openai_api_key
    )


gemini_api_key = st.secrets["GEMINI_API_KEY"]

genai.configure(
    api_key=gemini_api_key
)
#Name of models used
OPENAI_MODEL = "gpt-5.6-terra"

GEMINI_MODEL = "gemini-3-flash-preview"
#To read webpages
url_content = ""


if url1:

    content1 = read_url_content(url1)

    if content1:

        url_content += (
            "CONTENT FROM URL 1:\n"
            + content1
        )


if url2:

    content2 = read_url_content(url2)

    if content2:

        url_content += (
            "\n\nCONTENT FROM URL 2:\n"
            + content2
        )

#Step 4 System Prompt that is never discarded 
system_prompt_text = (
    "You are a helpful chatbot. "
    "Use the webpage content below as context when answering "
    "the user's questions. "

    "Answer questions clearly and in simple language. "

    "After answering a question, ask: "
    "'Do you want more info?' "

    "If the user says yes, provide more information and ask again: "
    "'Do you want more info?' "

    "If the user says no, ask: "
    "'What can I help you with?' "

    "\n\nWEBPAGE CONTEXT:\n"
    + url_content
)


system_message = {
    "role": "system",
    "content": system_prompt_text
}
#To initialize chat history
if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "How can I help you?"
        }
    ]

#To show full chat history
for message in st.session_state.messages:

    chat_message = st.chat_message(
        message["role"]
    )

    chat_message.write(
        message["content"]
    )
#Only show chat input if url was recieved

if not url1:

    st.info(
        "Enter at least one webpage URL in the sidebar to begin."
    )

else: #To get user input

    if prompt := st.chat_input("Ask a question about the webpage"):


        #Save user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )


        #Display user message
        with st.chat_message("user"):

            st.markdown(prompt)
     #Step 5:6 message conversation buffer
        recent_messages = []


        #Look through saved chat history
        for message in st.session_state.messages:

             #Only user and assistant messages in conversation buffer
            
            if (
                message["role"] == "user"
                or message["role"] == "assistant"
            ):

                recent_messages.append(message)


        #Keep only the last 6 messages
        if len(recent_messages) > 6:

            recent_messages = recent_messages[-6:]
        messages_to_send = (
            [system_message]
            + recent_messages
        )
        #OPENAI
        if llm_choice == "OpenAI":

            client = st.session_state.openai_client

            stream = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages_to_send,
                stream=True
            )


            with st.chat_message("assistant"):

                response = st.write_stream(stream)


        #GEMINI
        elif llm_choice == "Gemini":

            gemini_model = genai.GenerativeModel(
                model_name=GEMINI_MODEL,
                system_instruction=system_prompt_text
            )
            #Change recent messages into text
            #for gemini to receive the conversation history
            conversation_text = ""

            for message in recent_messages:

                conversation_text += (
                    message["role"]
                    + ": "
                    + message["content"]
                    + "\n"
                )


            gemini_stream = gemini_model.generate_content(
                conversation_text,
                stream=True
            )

            #Generator used by st.write_stream()
            def gemini_response():

                for chunk in gemini_stream:

                    if chunk.text:

                        yield chunk.text


            with st.chat_message("assistant"):

                response = st.write_stream(
                    gemini_response()
                )
        #To save assiatant response

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )