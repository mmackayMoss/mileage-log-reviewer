import streamlit as st
import tempfile
import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

from langchain_community.document_loaders import UnstructuredExcelLoader
import nltk
from bs4 import BeautifulSoup
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import Runnable

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']


def create_vector_store(vector_files: list[str]) -> InMemoryVectorStore:
    pages = []

    for file_path in vector_files:
        loader = PyPDFLoader(file_path)
        for page in loader.lazy_load():
            pages.append(page)

    return InMemoryVectorStore.from_documents(pages, OpenAIEmbeddings())


def create_rag_chain(vector_files: list[str]) -> Runnable:
    retriever = create_vector_store(vector_files).as_retriever(search_kwargs={"k": 1}) # Only retrieve 1 document since we are expecting a single file of truth
    model = ChatOpenAI(model="gpt-4")
    parser = StrOutputParser()
    
    system_prompt = (
        "You are an assistant for reviewing mileage expense logs. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Only respond with the mileage number AND NOTHING ELSE."
        "For example, if the mileage between the two locations is 63.5 miles, "
        "your response should be '63.5'."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(model, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

# Parses the Excel file and returns a list of errors
def parse_excel_file(excel_path: str, vector_files: list[str]) -> list[str]:
    errors = []

    loader = UnstructuredExcelLoader(excel_path, mode="elements")
    docs = loader.load()

    rag_chain = create_rag_chain(vector_files)
    
    for doc in docs:
        if 'category' in doc.metadata and doc.metadata['category'] == 'Table':
            if 'text_as_html' in doc.metadata:
                soup = BeautifulSoup(doc.metadata['text_as_html'], 'html.parser')
                first_cell = soup.find('td')
                if first_cell.string == 'Date':
                    # Main table starts here
                    data_rows = soup.find_all('tr')[1:]
                    
                    for index, data_row in enumerate(data_rows):
                        # Each row has 10 cells
                        # 1: To Location
                        # 3: From Location
                        # 8: Miles
                        cells = data_row.find_all('td')
                        if len(cells) >= 9: # Minimum number of cells for checks to work
                            to_loc = cells[1].string
                            from_loc = cells[3].string
                            miles = cells[8].string

                            # Check if required data is missing
                            if to_loc is None or from_loc is None or miles is None:
                                # Missing data in row
                                if to_loc is None and from_loc is None and miles is None:
                                    # Empty row, ignore
                                    continue
                                elif to_loc is None:
                                    errors.append(f"Missing 'To Location' in row {index + 1}")
                                elif from_loc is None:
                                    errors.append(f"Missing 'From Location' in row {index + 1}")
                                elif miles is None:
                                    errors.append(f"Missing 'Miles' in row {index + 1}")

                                continue

                            # Check corresponding locations for mileage
                            search_result = rag_chain.invoke({"input": f"What is the mileage amount between {to_loc} and {from_loc}?"})
                            answer = search_result['answer']
                            try:
                                answer = round(float(answer), 2)
                                if answer != round(float(miles), 2):
                                    errors.append(f"Mileage value in row {index + 1} does not match expected value of {answer}")
                            except:
                                errors.append(f"Could not retrieve expected mileage for row {index + 1}")

    return errors


st.title("Mileage Expense Log Reviewer")

uploaded_files = st.file_uploader("Upload your mileage log file and expected mileage files", type=["xlsx", "pdf"], accept_multiple_files=True)

if st.button("Validate Mileage"):
    if not uploaded_files:
        st.error("Please upload at least one file")
    else:
        # Save uploaded files to session state
        vector_files = []
        excel_path = None
        for file in uploaded_files:
            file_suffix = '.xlsx' if file.type != "application/pdf" else '.pdf'
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
                tmp_file.write(file.getvalue())
                if file_suffix == '.pdf':
                    vector_files.append(tmp_file.name)
                else:
                    excel_path = tmp_file.name
        
        if excel_path is None:
            st.error("Please upload the mileage log excel file")
        elif len(vector_files) == 0:
            st.error("At least one expected mileage file is required")
        else:
            errors = parse_excel_file(excel_path, vector_files)

            # Display results
            if errors:
                st.error("Errors found:")
                for error in errors:
                    st.write(f"- {error}")
            else:
                st.success("No mileage errors found!")

            # Clean up temporary files
            for file in vector_files:
                os.remove(file)
            if excel_path:
                os.remove(excel_path)
