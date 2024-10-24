__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.callbacks import get_openai_callback
from helper_functions.utility import check_password
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
import tiktoken

# Check if the password is correct.  
if not check_password():  
    st.stop()

# Sidebar contents
with st.sidebar:
    st.title('WSHO Assessment Assistant!')

def count_tokens(text):
    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
    return len(encoding.encode(text))

def main():
    load_dotenv()

    # Main Content
    st.header("Assessment Assistant for Workplace Safety and Health Officer (WSHO) Applicants")

    # Upload file
    pdf = st.file_uploader("Upload the document for WSH regulations and input the applicants' responses.", type="pdf")

    # Extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)  # Directly reading the PDF file
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()  # Extract text from each page and concatenate

        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)  # Split text into manageable chunks

        # Create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # Show user input
        with st.chat_message("user"):
            st.write("Document uploaded successfully!")

        # User input for questions or responses
        user_question = st.text_area("Please input your response or question about the uploaded file below:")
        if st.button("Submit"):
            if user_question:
                # Validate user's response with the document details
                docs = knowledge_base.similarity_search(user_question)

                # Define a custom prompt to validate user responses
                custom_prompt = """
                You are a safety regulations expert. Please validate the following response based on the given documents.
                Check if the user's response is correct and identify any details missing from the user's response.
                Response: {question}
                """

                prompt_template = PromptTemplate(template=custom_prompt, input_variables=["question"])

                # Use the embeddings model to create a Chroma database
                #embeddings_model = OpenAIEmbeddings(model='text-embedding-ada-002')
                #db = Chroma.from_texts(chunks, embeddings_model)

                # Set up the RetrievalQA chain
                chain = RetrievalQA.from_chain_type(
                    ChatOpenAI(model='gpt-3.5-turbo'),
                    retriever=knowledge_base.as_retriever(),
                    return_source_documents=True,  # Make inspection of document possible
                    chain_type_kwargs={"prompt": prompt_template, "document_variable_name": "documents"}
                )

                # Use 'invoke' instead of 'run'
                response = chain.invoke({"input_documents": docs, "question": user_question})

                st.write(response)

if __name__ == '__main__':
    main()
