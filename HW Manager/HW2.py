import streamlit as st
from openai import OpenAI, AuthenticationError
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai

#Get API keys from Streamlit secrets
open_api_key = st.secrets["OPEN_API_KEY"]
gemini_api_key = st.secrets["GEMINI_API_KEY"]

#Create OpenAI client
client = OpenAI(api_key=open_api_key)

#Gemini key
genai.configure(api_key=gemini_api_key)


#Read and extract text from the URL
def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        return soup.get_text()

    except requests.RequestException as e:
        print(f"Error reading {url}: {e}")
        return None


#For title and description
st.title("My URL Summarizer")

st.write(
    "Enter a webpage URL below and choose how you want the webpage summarized."
)


#The user enter a URL
url = st.text_input("Enter a webpage URL:")


#Summary options
summary_type = st.sidebar.radio(
    ":red[Choose a summary type:]",
    [
        "100 words",
        "Two paragraphs",
        "Five bullet points"
    ]
)


#Output languages
output_language = st.sidebar.selectbox(
    "Choose output language:",
    ["English", "Spanish", "French"]
)


#LLM selection
llm_choice = st.sidebar.selectbox(
    "Choose an LLM:",
    ["OpenAI", "Gemini"]
)


#Advanced model selection
advanced_model = st.sidebar.checkbox("Use advanced model")


#Create the summary instructions
if summary_type == "100 words":
    summary_instruction = "Summarize this webpage in 100 words."

elif summary_type == "Two paragraphs":
    summary_instruction = "Summarize this webpage in 2 connecting paragraphs."

else:
    summary_instruction = "Summarize this webpage in 5 bullet points."


#Add the selected language to the prompt
summary_instruction += f" Write the summary in {output_language}."



if llm_choice == "OpenAI":

    if advanced_model:
        model_to_use = "gpt-5.6-terra"
    else:
        model_to_use = "gpt-5.6-luna"


elif llm_choice == "Gemini":

    if advanced_model:
        model_to_use = "gemini-3-flash-preview"
    else:
        model_to_use = "gemini-3.1-flash-lite"


#Read webpage after user enters a URL
if url:

    document = read_url_content(url)

    if document:

        prompt = f"""
        Here is the webpage content:

        {document}

        {summary_instruction}
        """


        #Step 11:if OpenAI is selected
        if llm_choice == "OpenAI":

            try:
                client.models.list()

            except AuthenticationError:
                st.error("Incorrect OpenAI API key.")
                st.stop()


            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]


            stream = client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                stream=True
            )


            st.write_stream(stream)


        #If Gemini is selected
        elif llm_choice == "Gemini":

            try:
                list(genai.list_models())

            except Exception:
                st.error("Incorrect Gemini API key.")
                st.stop()


            gemini_model = genai.GenerativeModel(model_to_use)

            response = gemini_model.generate_content(prompt)

            st.write(response.text)

