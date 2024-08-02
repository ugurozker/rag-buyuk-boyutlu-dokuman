```markdown
# Ziraat Bank Question-Answering System

## Overview

This project implements a sophisticated question-answering system for Ziraat Bank using Retrieval-Augmented Generation (RAG) with Milvus Vector Database and WatsonX.ai.
It processes PDF documents, creates embeddings, performs similarity searches, and generates accurate answers to queries using GenAI capabilities of Watsonx.ai platform.

## Features

- Document loading and text chunking from PDF files
- Vector embedding creation using HuggingFaceInstructEmbeddings
- Similarity search using Milvus Vector Database
- Question answering using WatsonX.ai language model
- Cross-encoder for re-ranking search results
- Batch processing of questions from Excel files

## Prerequisites and Installation

1. Ensure you have Python 3.x installed.
2. Clone the repository and navigate to the project directory.
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. Create a `config.ini` file in the project root:
   ```ini
   [DEFAULT]
   ModelID = your_model_id
   EmbeddingsModel = sentence-transformers/paraphrase-multilingual-mpnet-base-v2

   [Milvus]
   CollectionName = your_collection_name
   Host = your_milvus_host
   Port = your_milvus_port
   User = your_milvus_user
   Password = your_milvus_password
   ServerPemPath = path_to_server_pem
   ServerName = your_server_name
   ```

2. Create a `.env` file in the project root:
   ```ini
   API_KEY=your_watsonx_api_key
   IBM_CLOUD_URL=your_ibm_cloud_url
   PROJECT_ID=your_project_id
   ```

## Usage

### Single Query

```python
query = "Your question here"
folder_path = 'Input'
config_path = 'config.ini'

ziraat_bank_qa = ZiraatBankQA(config_path)
response, context = ziraat_bank_qa.main(query, folder_path)
print(response)
```

### Batch Processing

```python
import pandas as pd

excel_file_path = 'Validation_set/validation_set.xlsx'
df = pd.read_excel(excel_file_path, engine='openpyxl')

ziraat_bank_qa = ZiraatBankQA('config.ini')
responses = []

for query in df['Soru']:
    response, context = ziraat_bank_qa.main(query, 'Input')
    responses.append(response)

df['response'] = responses
df.to_csv("Validation_Results_v1.csv")
```

## How It Works

- **Document Loading**: PDF files are loaded and split into chunks.
- **Embedding Creation**: Text chunks are converted into vector embeddings.
- **Vector Store**: Embeddings are stored in Milvus Vector Database.
- **Query Processing**:
  - Similarity search finds relevant text chunks.
  - Cross-encoder re-ranks the results.
  - Top relevant chunks form the context.
- **Answer Generation**: WatsonX.ai generates the final answer using the context and query.

## Customization

- Adjust `chunk_size` and `chunk_overlap` in `RecursiveCharacterTextSplitter` for different text splitting strategies.
- Modify `GenParams` in `send_to_watsonxai` method to tune the language model's output.
- Change the embeddings model in the config file.

## Troubleshooting

- Ensure all credentials in `config.ini` and `.env` are correct.
- Check that the Milvus server is running and accessible.
- Verify permissions for WatsonX.ai services.
```

Feel free to refer this `README.md` file for the project. Make sure to replace placeholders like `your_model_id`, `your_collection_name`, `your_milvus_host`, `your_milvus_port`, `your_milvus_user`, `your_milvus_password`, `path_to_server_pem`, `your_server_name`, `your_watsonx_api_key`, `your_ibm_cloud_url`, and `your_project_id` with your actual project details.
