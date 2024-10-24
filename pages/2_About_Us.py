import streamlit as st

# region <--------- Streamlit App Configuration --------->
st.set_page_config(
    layout="centered",
    page_title="My Streamlit App"
)

# endregion <--------- Streamlit App Configuration --------->

st.title("About Us")

st.markdown("""
## Project Scope
The scope of this project is to assist MOM officers in the Licensing Department to summarise and validate the responses from WSHO Applicants. 
The MOM Licensing branch, operating under PICS (Planning, Info & Corp Services) within OSHD, is responsible for evaluating and approving applications for the registrations of WSHOs 
(Workplace Safety and Health Officer). As part of the assessment criteria, applicants are required to complete the Professional Work Review Write-up Form.")

## Objective
In the process of assessing an applicant’s suitability, officers from the Licensing Branch will have to manually review and evaluate “Section E: Demonstration of WSH Experience” within the assessment form. 

In this section, applicants are required to demonstrate their WSH experience and knowledge by accurately citing examples of Section/Regulation No. for the following WSH legislations. 

The objective of this App is to assist the officers to efficiently assess the Workplace Safety and Health Officer (WSHO) Applicants by summarising applicants’ responses.

## Available Data

The data used are the regulation documents containing the WSH legislations. 
Below are a few examples of the Regulatory documents: 
- Workplace Safety and Health (General Provisions) Regulations: https://sso.agc.gov.sg/SL/WSHA2006-RG1.
- Workplace Safety and Health (Incident Reporting) Regulations: https://sso.agc.gov.sg/SL-Supp/S735-2020/Published/20200831?DocDate=20200831         

## Features

- Summarize the inputs provided. 
- Validate the details of the inputs are factual and correct.       
""")