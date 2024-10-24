from dotenv import load_dotenv
import streamlit as st
import time
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from helper_functions.utility import check_password  

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

                # Modify the prompt to not only check but also validate any missing details
                llm = OpenAI()
                chain = load_qa_chain(llm, chain_type="stuff")
                validation_prompt = (
                    f"Check if the user's response is correct based on the document, and identify if there are any "
                    f"details missing from the user's response. Response: '{user_question}'"
                )
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=validation_prompt)
                    print(cb)

if __name__ == '__main__':
    main()