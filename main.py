from dotenv import load_dotenv
import streamlit as st
import time
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.callbacks import get_openai_callback
from helper_functions.utility import check_password
from langchain.chains import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import tiktoken
from langchain.document_loaders import PyPDFLoader

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

    #Main Content
    st.header("Assessment assistant for Workplace Safety and Health Officer (WSHO) Applicants")

    # upload file
    pdf = st.file_uploader("Upload the document for WSH regulations and input the applicants' responses.", type="pdf")
    
    pages = pdf.load()

    # extract the text
    if pdf is not None:
      pdf_reader = PdfReader(pdf)
      text = ""
      for page in pdf_reader.pages:
        text += page.extract_text()
        
      # split into chunks
      text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
      chunks = text_splitter.split_text(text)
      
      # create embeddings
      embeddings = OpenAIEmbeddings()
      knowledge_base = FAISS.from_texts(chunks, embeddings)
      
      # show user input
      with st.chat_message("user"):
        st.write("Document uploaded successfuly!")

    
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

                # Load the document, split it into chunks, embed each chunk and load it into the vector store.
                text_splitter = RecursiveCharacterTextSplitter(
                    separators=["\n\n", "\n", " ", ""],
                    chunk_size=500,
                    chunk_overlap=50,
                    length_function=count_tokens
                    )

                splitted_documents = text_splitter.split_documents(pages)

                embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small')
                db = Chroma.from_documents(splitted_documents, embeddings_model, persist_directory="./chroma_db")
                chain = RetrievalQA.from_chain_type(
                    ChatOpenAI(model='gpt-3.5-turbo'),
                    retriever=db.as_retriever(),
                    return_source_documents=True, # Make inspection of document possible
                    chain_type_kwargs={"prompt": prompt_template})

                # Use 'invoke instead of 'run'
                response = chain.invoke({"input_documents": docs, "question": user_question})
            
                st.write(response)

if __name__ == '__main__':
    main()