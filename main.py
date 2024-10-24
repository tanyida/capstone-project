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
from langchain_openai import OpenAI

# Check if the password is correct.  
if not check_password():  
    st.stop()
    
# Sidebar contents
with st.sidebar:
    st.title('WSHO Assessment Assistant!')

def main():
    load_dotenv()

    #Main Content
    st.header("Assessment assistant for Workplace Safety and Health Officer (WSHO) Applicants")

    # upload file
    pdf = st.file_uploader("Upload the document for WSH regulations and input the applicants' responses.", type="pdf")
    
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
                
                llm = OpenAI()
                chain = StuffDocumentsChain(llm=llm, prompt=prompt_template)

                # Use 'invoke instead of 'run'
                response = chain.invoke({"input_documents": docs, "question": user_question})
            
                st.write(response)

if __name__ == '__main__':
    main()