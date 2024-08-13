import os,io,time,shutil,copy
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import uvicorn
import pdfplumber
from configparser import ConfigParser
from pymilvus import connections, Collection, utility
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Milvus
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from prompt import prompt_generator,excel_text
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceInstructEmbeddings
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List, Literal , Optional, Dict
from pydantic import BaseModel, validator
from deep_search import DeepSearchExtraction,extract_tables
import deep_search as ds  # Ensure this import matches your actual module
from typing import List, Dict
from docx import Document

def process_and_extract_table(file_path: str):
    # Initialize the DeepSearchExtraction with the output directory
    deep_search_extractor = DeepSearchExtraction(output_dir="Output")

    # Process the file
    deep_search_extractor.process(file_path)

    # Extract the required data
    extracted_data = deep_search_extractor.extract(extract_tables=True, extract_text=True, visualise_table=True)

    # Extract tables from the extracted data
    df_table_list = extract_tables(extracted_data)

    # Merge data frames
    merged_df = pd.DataFrame()
    for data in df_table_list:
        merged_df = pd.concat([merged_df, data], ignore_index=True)

    # Convert the data frame to a text representation
    columns_as_lists = merged_df.fillna("ilgili").to_dict(orient='list')
    text_representation = "\n".join(",".join(map(str, col)) for col in columns_as_lists.values())

    return text_representation

# ### RAG- using Milvus Vector DB

def docx_to_text(file: io.BytesIO) -> str:
    doc = Document(file)
    text = []
    for para in doc.paragraphs:
        text.append(para.text)
    return '\n'.join(text)

def table_analysis(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        if len(pdf.pages) > 10:
            raise HTTPException(status_code=500, detail=f"Could not retrieve PDF page count from maximum 10")
        text = "".join([page.extract_text() for page in pdf.pages])
        for page in pdf.pages:
            tables = page.extract_tables()
            if tables:
                return True, text

    return False,text


class ZiraatBankQA:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)

        self.creds ,self.project_id = self.get_wml_creds()
        self.model_id = self.config['DEFAULT']['ModelID']
        self.embeddings = HuggingFaceInstructEmbeddings(model_name=self.config['DEFAULT']['EmbeddingsModel'])

        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ' '],
            chunk_size=700,
            chunk_overlap=100,
            length_function=len
        )
        # self.collection_name = self.config['Milvus']['CollectionName']
        self.host = self.config['Milvus']['Host']
        self.port = self.config['Milvus']['Port']
        self.user = self.config['Milvus']['User']
        self.password = self.config['Milvus']['Password']
        self.server_pem_path = self.config['Milvus']['ServerPemPath']
        self.server_name = self.config['Milvus']['ServerName']

    def load_config(self, config_path):
        config = ConfigParser()
        config.read(config_path)
        return config

    def get_wml_creds(self):
        api_key = "vyGwoIw0BSLZnthc5TLWUI3PVxO24miIhdGyJH_NS2ll"
        ibm_cloud_url = "https://us-south.ml.cloud.ibm.com"
        project_id = "86cc43a6-c2f0-4e3e-a6e9-426ac8cf8f7b"
        if api_key is None or ibm_cloud_url is None or project_id is None:
            print("Ensure you copied the .env file that you created earlier into the same directory as this script")
        else:
            creds = {
                "url": ibm_cloud_url,
                "apikey": api_key
            }
        return creds ,project_id

    def send_to_watsonxai(self, prompt, max_token=200):
        params = {
            GenParams.DECODING_METHOD: "greedy",
            GenParams.MIN_NEW_TOKENS: 1,
            GenParams.MAX_NEW_TOKENS: max_token,
            GenParams.TEMPERATURE: 0,
        }
        model = Model(model_id=self.model_id, params=params, credentials=self.creds, project_id=self.project_id)
        response = model.generate_text(prompt)
        return response

    def load_documents(self, folder_path):
        text_chunks = []
        files = glob.glob(os.path.join(folder_path, '*.pdf'))

        for file in tqdm(files):
            with pdfplumber.open(file) as pdf:
                data = ''.join([page.extract_text() for page in pdf.pages])

            created_text_chunks = self.text_splitter.create_documents([data])
            for chunks in created_text_chunks:
                chunks.metadata['file'] = file
                text_chunks.append(chunks)

        return text_chunks

    def util_connection(self,):

        connections.connect(
            "default",
            host=self.host,
            port=self.port,
            secure=True,
            server_pem_path=self.server_pem_path,
            server_name=self.server_name,
            user=self.user,
            password=self.password
        )

    def create_vector_store(self, text_chunks,collection_name):
        self.util_connection()

        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)

        vector_db = Milvus.from_documents(
            text_chunks,
            self.embeddings,
            connection_args={
                "host": self.host,
                "port": self.port,
                "secure": True,
                "server_pem_path": self.server_pem_path,
                "server_name": self.server_name,
                "user": self.user,
                "password": self.password
            },
            collection_name=collection_name
        )

        collection = Collection(collection_name)
        collection.load()
        return vector_db,collection

    def get_vector_store(self, collection_name):
        self.util_connection()

        if not utility.has_collection(collection_name):
            raise HTTPException(status_code=500, detail=f"Collection name is not included.")


        vector_db = Milvus(embedding_function= self.embeddings,
            connection_args={
                "host": self.host,
                "port": self.port,
                "secure": True,
                "server_pem_path": self.server_pem_path,
                "server_name": self.server_name,
                "user": self.user,
                "password": self.password
            },
            collection_name=collection_name
        )
        return vector_db

    def perform_qa(self, df, query,max_token, collection_name):
        context = "\n\n".join(df['paragraph'])
        prompt = prompt_generator( context, query, collection_name)
        response = self.send_to_watsonxai(prompt,max_token)
        return response, context

    def create_model(self, model_name):
        model = CrossEncoder(model_name, max_length=512)
        return model
    

    def main(self, query, vector_db, model, max_token=200, collection_name= "general"):
        docs = vector_db.similarity_search_with_score(query, k=12, ef=7)

        _docs = pd.DataFrame(
            [(query, doc[0].page_content, doc[0].metadata.get('file'), doc[1]) for doc in docs],
            columns=['query', 'paragraph', 'document', 'relevent_score']
        )
        scores = model.predict(_docs[['query', 'paragraph']].to_numpy())
        _docs['score'] = scores
        df = _docs[:12]

        response, context = self.perform_qa(df, query, max_token, collection_name)
        return response, context

app = FastAPI()


try:
    # Initialize embeddings
    embeddings = HuggingFaceInstructEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    model_kwargs={"device": "cpu"}
    )
except Exception as e:
    raise HTTPException(status_code=500, detail=f"HuggingFaceInstructEmbeddings couldn't fetch from HF: {str(e)}")

try:
    config_path = 'config.ini'
    ziraat_bank_qa = ZiraatBankQA(config_path)
    model = ziraat_bank_qa.create_model('emrecan/bert-base-turkish-cased-mean-nli-stsb-tr')
    ziraat_bank_qa.util_connection()
    current_collection = "general"
    vector_db = ziraat_bank_qa.get_vector_store(current_collection)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error performing __main__ operation: {str(e)}")

# import uvicorn
# uvicorn.run(app, host="0.0.0.0", port=8000)

class UploadResponse(BaseModel):
    query_response: str
    collection_name: str
    response_duration: str

@app.post("/upload_pdfs", response_model=UploadResponse)
async def upload_pdfs(collection_name: str, files: List[UploadFile]):
    start_time = time.time()
    # Validate number of files
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="You can upload a maximum of 10 PDF files at a time.")

    # Load and process each PDF file
    text_chunks = []
    try:
        for file in files:
            if file.content_type == "application/pdf":
                table_content = ""
                is_DeepSearch, text = table_analysis(file.file)
                try:
                    if is_DeepSearch :
                        # Dosyayı geçici bir dosyaya kaydedin
                        with open(f"/tmp/{file.filename}", "wb") as buffer:
                            shutil.copyfileobj(file.file, buffer)

                        # İşleme fonksiyonunu çağırın
                        table_content = process_and_extract_table(f"/tmp/{file.filename}")
                        text = text + " \n " + table_content
                        # Geçici dosyayı silin
                        os.remove(f"/tmp/{file.filename}")
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Error processing PDF file {file.filename}: {str(e)}")

            elif file.content_type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
                excel_content = await file.read()
                try:
                    # Read the Excel file
                    df = pd.read_excel(io.BytesIO(excel_content),decimal='.')
                    # Convert the DataFrame to text
                    text = str(df.to_dict(orient="records"))
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Error processing Excel file {file.filename}: {str(e)}")
            

            elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                doc_content = await file.read()
                try:
                    text = docx_to_text(io.BytesIO(doc_content))
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Error processing DOCX file {file.filename}: {str(e)}")
            
            else:
                raise HTTPException(status_code=400, detail=f"File {file.filename} is neither a PDF nor an Excel file.")

            try:
                text_chunks.extend(ziraat_bank_qa.text_splitter.create_documents([text]))
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error splitting text into chunks: {str(e)}")

        # Create a vector store in Milvus
        try:
            vector_db, collection = ziraat_bank_qa.create_vector_store(text_chunks, collection_name)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error creating vector store: {str(e)}")

        # query = "1/7/2023 tarihinde Yabancı Para cinsinden Kredi Kartı işlemlerinde uygulanacak azami gecikme faiz oranı yüzde kaçtır?"
        # q_response = vector_db.similarity_search(query)
        response_text = " , ".join([file.filename for file in files]) + " named files " + str(utility.load_state(collection_name)) + " to Milvus " + collection_name +" collection"
        duration_text = str(time.time() - start_time)+" seconds"
        return UploadResponse(query_response=response_text, collection_name=collection_name, response_duration=duration_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"General error processing files: {str(e)}")

class BigRAGRequest(BaseModel):
    query: str
    max_token: int

class BigRAGResponse(BaseModel):
    collection_name: str
    query : str
    result: str

@app.post("/perform_big_rag", response_model=BigRAGResponse)
async def perform_big_rag(rag_request: BigRAGRequest):
    try:
        collection_name = "ziraat_big_pdf"
        query = rag_request.query
        global current_collection,vector_db
        # Ensure the vector store is initialized appropriately
        try:
            if collection_name != current_collection:
                vector_db = ziraat_bank_qa.get_vector_store(collection_name)
                current_collection = collection_name
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error initializing vector store: {str(e)}")

        #model = ziraat_bank_qa.create_model('emrecan/bert-base-turkish-cased-mean-nli-stsb-tr')

        try:
            big_ziraat_object = copy.copy(ziraat_bank_qa)
            big_ziraat_object.model_id = 'meta-llama/llama-3-70b-instruct'
            response, _ = big_ziraat_object.main(query, vector_db, model,rag_request.max_token, collection_name)        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error performing main operation: {str(e)}")

        return BigRAGResponse(collection_name=collection_name, query=query, result=response)

    except HTTPException as http_ex:
        raise http_ex

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

#--------------------------------------------------------------------------------------------------

class RevisedRequest(BaseModel):
    query: str
    max_token: int

class RevisedResponse(BaseModel):
    collection_name: str
    query : str
    result: str

@app.post("/perform_revised_rag", response_model=RevisedResponse)
async def perform_revised_rag(rag_request: RevisedRequest):
    try:
        collection_name = "ziraat_revised_pdf"
        query = rag_request.query
        global current_collection,vector_db
        # Ensure the vector store is initialized appropriately
        try:
            if collection_name != current_collection:
                vector_db = ziraat_bank_qa.get_vector_store(collection_name)
                current_collection = collection_name
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error initializing vector store: {str(e)}")

        #model = ziraat_bank_qa.create_model('emrecan/bert-base-turkish-cased-mean-nli-stsb-tr')

        try:
            response, _ = ziraat_bank_qa.main(query, vector_db, model, rag_request.max_token, collection_name)        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error performing main operation: {str(e)}")

        return RevisedResponse(collection_name=collection_name, query=query, result=response)

    except HTTPException as http_ex:
        raise http_ex

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    
#--------------------------------------------------------------------------------------------------

class RefereRequest(BaseModel):
    query: str
    max_token: int

class RefereResponse(BaseModel):
    collection_name: str
    query : str
    result: str

@app.post("/perform_refere_rag", response_model=RefereResponse)
async def perform_refere_rag(rag_request: RefereRequest):
    try:
        collection_name = "ziraat_refere_pdf"
        query = rag_request.query
        global current_collection,vector_db
        # Ensure the vector store is initialized appropriately
        try:
            if collection_name != current_collection:
                vector_db = ziraat_bank_qa.get_vector_store(collection_name)
                current_collection = collection_name
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error initializing vector store: {str(e)}")

        #model = ziraat_bank_qa.create_model('emrecan/bert-base-turkish-cased-mean-nli-stsb-tr')

        try:
            response, _ = ziraat_bank_qa.main(query, vector_db, model, rag_request.max_token, collection_name)        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error performing main operation: {str(e)}")

        return RefereResponse(collection_name=collection_name, query=query, result=response)

    except HTTPException as http_ex:
        raise http_ex

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
#--------------------------------------------------------------------------------------------------

class ExcelRequest(BaseModel):
    query: str
    max_token: int

class ExcelResponse(BaseModel):
    collection_name: str
    query : str
    result: str

@app.post("/perform_excel_rag", response_model=ExcelResponse)
async def perform_excel_rag(rag_request: ExcelRequest):
    try:
        collection_name = "ziraat_excel"
        query = rag_request.query
        global current_collection,vector_db
        # Ensure the vector store is initialized appropriately
        try:
            if collection_name != current_collection:
                vector_db = ziraat_bank_qa.get_vector_store(collection_name)
                current_collection = collection_name
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error initializing vector store: {str(e)}")

        #model = ziraat_bank_qa.create_model('emrecan/bert-base-turkish-cased-mean-nli-stsb-tr')

        try:
            big_ziraat_object = copy.copy(ziraat_bank_qa)
            big_ziraat_object.model_id = 'meta-llama/llama-3-70b-instruct'
            response, _ = big_ziraat_object.main(query, vector_db, model,rag_request.max_token, collection_name)  
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error performing main operation: {str(e)}")

        return ExcelResponse(collection_name=collection_name, query=query, result=response)

    except HTTPException as http_ex:
        raise http_ex

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
#--------------------------------------------------------------------------------------------------

class TablePDFRequest(BaseModel):
    query: str
    max_token: int

class TablePDFResponse(BaseModel):
    collection_name: str
    query : str
    result: str

@app.post("/perform_table_rag", response_model=TablePDFResponse)
async def perform_table_rag(rag_request: TablePDFRequest):
    try:
        collection_name = "ziraat_table_pdf"
        query = rag_request.query
        global current_collection,vector_db
        # Ensure the vector store is initialized appropriately
        try:
            if collection_name != current_collection:
                vector_db = ziraat_bank_qa.get_vector_store(collection_name)
                current_collection = collection_name
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error initializing vector store: {str(e)}")

        #model = ziraat_bank_qa.create_model('emrecan/bert-base-turkish-cased-mean-nli-stsb-tr')

        try:
            response, _ = ziraat_bank_qa.main(query, vector_db, model, rag_request.max_token, collection_name)          
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error performing main operation: {str(e)}")

        return TablePDFResponse(collection_name=collection_name, query=query, result=response)

    except HTTPException as http_ex:
        raise http_ex

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
#--------------------------------------------------------------------------------------------------

class RAGRequest(BaseModel):
    collection_name: str
    query: str
    max_token: int

    # @validator('collection_name')
    # def check_option(cls, v):
    #     if v not in utility.list_collections():
    #         raise ValueError(f"Invalid option: {v}. Must be one of {utility.list_collections()}.")
    #     return v

class RAGResponse(BaseModel):
    collection_name: str
    query : str
    result: str

@app.post("/perform_rag", response_model=RAGResponse)
async def perform_rag(rag_request: RAGRequest):

    try:
        query = rag_request.query
        global current_collection,vector_db
        # Ensure the vector store is initialized appropriately
        try:
            if rag_request.collection_name != current_collection:
                vector_db = ziraat_bank_qa.get_vector_store(rag_request.collection_name)
                current_collection = rag_request.collection_name
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error initializing vector store: {str(e)}")

        #model = ziraat_bank_qa.create_model('emrecan/bert-base-turkish-cased-mean-nli-stsb-tr')

        try:
            response, _ = ziraat_bank_qa.main(query, vector_db, model, rag_request.max_token, rag_request.collection_name)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error performing main operation: {str(e)}")

        return RAGResponse(collection_name=rag_request.collection_name, query=query, result=response)

    except HTTPException as http_ex:
        raise http_ex

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

#--------------------------------------------------------------------------------------------------

class BulkRAGRequest(BaseModel):
    collection_name: str
    query: str
    file: Optional[UploadFile] = None

    @validator('collection_name')
    def check_option(cls, v):
        if v not in utility.list_collections():
            raise ValueError(f"Invalid option: {v}. Must be one of {utility.list_collections()}.")
        return v

class BulkRAGResponse(BaseModel):
    collection_name: str
    query : str
    result_list: List[dict] = []


@app.post("/perform_bulk_rag", response_model=BulkRAGResponse)
async def perform_bulk_rag(query: str,
    collection_name: str, files: List[UploadFile]):
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="You can upload a maximum of 10 files at a time.")
    global current_collection,vector_db
    # Ensure the vector store is initialized appropriately
    try:
        if collection_name != current_collection:
            vector_db = ziraat_bank_qa.get_vector_store(collection_name)  # Assuming vector_db has been initialized appropriately
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing vector store: {str(e)}")

    # model = ziraat_bank_qa.create_model('emrecan/bert-base-turkish-cased-mean-nli-stsb-tr')

    bulk_list = []

    if files:
        for file in files:
            if file.content_type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
                try:
                    contents = await file.read()
                    df = pd.read_excel(io.BytesIO(contents))

                    queries = df.iloc[:, 0].tolist()

                    for index_query in queries:
                        try:
                            response, _ = ziraat_bank_qa.main(index_query, vector_db, model)
                            bulk_list.append({"query": index_query, "response": response})
                        except Exception as e:
                            bulk_list.append({"query": index_query, "response": f"Error processing query: {str(e)}"})
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Error processing Excel file {file.filename}: {str(e)}")
    else:
        try:
            responses, _ = ziraat_bank_qa.main(query, vector_db, model, collection_name=collection_name)
            bulk_list.append({"query": query, "response": responses})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error performing bulk RAG operation: {str(e)}")

    return BulkRAGResponse(collection_name=collection_name, query=query, result_list=bulk_list)

#-----------------------------------------------------

@app.get("/collections", response_model=List[str])
async def get_collections():
    try:
        collections = utility.list_collections()
        return collections
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving collections: {str(e)}")


class DropCollectionRequest(BaseModel):
    collection_name: str

@app.post("/drop_collection")
async def drop_collection(request: DropCollectionRequest):
    try:
        if request.collection_name in ["ziraat_bank","ziraat_bank_large","Ziraat_Col","ziraat_big_pdf","ziraat_excel","ziraat_table_pdf","ziraat_bank_webscrape_index"]:
            result = request.collection_name + " main collections could not drop"
            return {"message": result}
        elif utility.has_collection(request.collection_name):
            utility.drop_collection(request.collection_name)
            result = request.collection_name + " dropped successfully"
            return {"message": result}
        else:
            result = request.collection_name + " couldn't drop or already dropped"
        return {"message": result}
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error dropping collection: {str(e)}")

# Run the app
# if __name__ == "__main__":
#     # Initialize ZiraatBankQA instance
#     try:
#         config_path = 'config.ini'
#         ziraat_bank_qa = ZiraatBankQA(config_path)
#         model = ziraat_bank_qa.create_model('emrecan/bert-base-turkish-cased-mean-nli-stsb-tr')
#         ziraat_bank_qa.util_connection()
#         uvicorn.run(app, host="0.0.0.0", port=8000)

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error performing __main__ operation: {str(e)}")



