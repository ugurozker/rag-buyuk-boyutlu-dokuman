import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from configparser import ConfigParser
from pymilvus import connections, Collection, utility
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Milvus
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from sentence_transformers import SentenceTransformer
import os
import json
import tempfile
import deepsearch as ds
import pandas as pd
import logging
import pexpect
from zipfile import ZipFile
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from IPython.display import display, Markdown, HTML
from io import StringIO
from typing import List, Dict, Optional

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('deepsearch_table_extraction.log')
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

class TableExtractor:
    @staticmethod
    def get_tablecell_span(cell, ix):
        span = set(s[ix] for s in cell['spans'])
        if len(span) == 0:
            return 1, None, None
        return len(span), min(span), max(span)

    @staticmethod
    def write_table(item):
        table = item
        body = "<table>\n"

        nrows = table['#-rows']
        ncols = table['#-cols']

        for i in range(nrows):
            body += "  <tr>\n"
            for j in range(ncols):
                cell = table['data'][i][j]
                rowspan, rowstart, rowend = TableExtractor.get_tablecell_span(cell, 0)
                colspan, colstart, colend = TableExtractor.get_tablecell_span(cell, 1)

                if rowstart is not None and rowstart != i: continue
                if colstart is not None and colstart != j: continue

                if rowstart is None:
                    rowstart = i
                if colstart is None:
                    colstart = j

                content = cell['text'] if cell['text'] else '&nbsp;'
                label = cell['type']
                label_class = 'header' if label in ['row_header', 'row_multi_header', 'row_title', 'col_header', 'col_multi_header'] else 'body'
                celltag = 'th' if label_class == 'header' else 'td'
                style = 'style="text-align: center;"' if label_class == 'header' else ''

                body += f'    <{celltag} rowstart="{rowstart}" colstart="{colstart}" rowspan="{rowspan}" colspan="{colspan}" {style}>{content}</{celltag}>\n'
            body += "  </tr>\n"

        body += "</table>"
        return body

    @staticmethod
    def extract_document_tables(doc_jsondata):
        pdf_tables = {}
        for table in doc_jsondata.get("tables", []):
            prov = table["prov"][0]
            page = prov["page"]
            pdf_tables.setdefault(page, [])
            output_html = TableExtractor.write_table(table)
            pdf_tables[page].append(output_html)
        return pdf_tables


class DeepSearchExtraction:
    def __init__(self, output_dir=None):
        self.output_dir = output_dir if output_dir else tempfile.mkdtemp()
        self.api = self.initialize_api()
        self.doc_jsondata = None
        self.pdf_data = {}

    def initialize_api(self):
        login_response = self.login()
        if login_response['status']:
            logger.info('Connection Established with DeepSearch')
            return ds.CpsApi.from_env()
        else:
            logger.error("DeepSearch Connection Issue", login_response['exception'])
            raise ConnectionError("Unable to establish connection with DeepSearch")

    def login(self):
        try:
            command = "deepsearch profile config --profile-name 'ds-experience' --host 'https://deepsearch-experience.res.ibm.com/' --no-verify-ssl --username 'ugur.ozker@ibm.com'"
            process = pexpect.spawn(command)

            api_key = "aZ06q4BX0vS5UxB-22JFIV4HkQyW2BKqeOliU22TYe4"
            process.expect("Api key:")
            process.sendline(api_key)
            process.expect(pexpect.EOF)
            return {"status": True}
        except Exception as exp:
            return {"status": False, "exception": exp}

    def display_table(self, tables_dict):
        for page, tables in tables_dict.items():
            for idx, table_html in enumerate(tables):
                display(Markdown(f"## Table {idx + 1} on page {page}"))
                display(HTML(table_html))

    def process(self, file_path):
        try:
            logger.info("Processing: %s", file_path)
            documents = ds.convert_documents(
                api=self.api,
                proj_key="1234567890abcdefghijklmnopqrstvwyz123456",
                source_path=file_path,
                progress_bar=True
            )

            documents.download_all(result_dir=self.output_dir, progress_bar=True)

            for output_file in Path(self.output_dir).rglob("json*.zip"):
                with ZipFile(output_file) as archive:
                    for name in archive.namelist():
                        if name.endswith(".json"):
                            basename = name.split(".json")[0]
                            self.doc_jsondata = json.loads(archive.read(f"{basename}.json"))
        except Exception as exp:
            logger.error("Processing Failed:", exp)

    def extract(self, extract_tables=False, extract_text=False, visualise_table=False):
        try:
            self.pdf_data = {
                "total_pages": self.doc_jsondata['file-info']['#-pages'],
                "data": {page_num: {"text": "", "tables": []} for page_num in range(1, self.doc_jsondata['file-info']['#-pages'] + 1)},
                "pdf_name": self.doc_jsondata['file-info']['filename']
            }

            if extract_text:
                for metadata in self.doc_jsondata['main-text']:
                    if 'text' in metadata:
                        page_no = int(metadata['prov'][0]['page'])
                        self.pdf_data['data'][page_no]['text'] += " \n" + metadata['text']

            if extract_tables:
                tables = TableExtractor.extract_document_tables(self.doc_jsondata)
                for page_no, table_html in tables.items():
                    self.pdf_data['data'][page_no]['tables'] = table_html

                if visualise_table:
                    self.display_table(tables)

            return self.pdf_data
        except Exception as exp:
            logger.error("Extraction Failed:", exp)



# %%
def html_to_df(html_string: str) -> Optional[pd.DataFrame]:
    """Convert an HTML string to a pandas DataFrame."""
    try:
        df = pd.read_html(StringIO(str(html_string)), decimal=',', thousands='.')[0]  # Assumes that there is only one table per HTML string
        return df
    except ValueError as e:
        print(f"ValueError: {e} - The HTML string might not contain a valid table.")
    except Exception as e:
        print(f"An error occurred while converting HTML to DataFrame: {e}")
    return None

def extract_tables(extracted_data: Dict[str, Dict[str, List[str]]]) -> List[pd.DataFrame]:
    """Extract tables from the provided data and return them as a list of pandas DataFrames."""
    df_table_list = []
    try:
        num_pages = set(extracted_data.get("data", {}).keys())
        if num_pages:
            for page in num_pages:
                print(f"Processing page: {page}")
                tables_list = extracted_data["data"].get(page, {}).get("tables", [])
                if tables_list:  # if not empty
                    for table_string in tables_list:
                        df = html_to_df(table_string)
                        if df is not None:
                            df_table_list.append(df)
    except KeyError as e:
        print(f"KeyError: {e} - Missing expected key in the extracted data.")
    except Exception as e:
        print(f"An error occurred while extracting tables: {e}")
    
    return df_table_list

if __name__ == "__main__":

    # %%
    # Create an instance of the DeepSearchExtraction class
    deep_search_extractor = DeepSearchExtraction(output_dir="Output")

    # Process the file
    file_path = "Large/AzamiFaizOrn.pdf"
    deep_search_extractor.process(file_path)

    # Extract the required data
    extracted_data = deep_search_extractor.extract(extract_tables=True, extract_text=True, visualise_table=True)

    # Now, you can use the extracted_data for further processing or analysis
    print(extracted_data)


    # %%
    df_table_list = extract_tables(extracted_data)

    # %%
    merged_df = pd.DataFrame()
    for data in df_table_list:
        merged_df = pd.concat([merged_df, data], ignore_index=True)

    text_data = merged_df.values.tolist()
    text_data_str = '\n'.join([','.join(map(str, row)) for row in merged_df.values])

    # %%
    columns_as_lists = merged_df.fillna("ilgili").to_dict(orient='list')
    text_representation = "\n".join(",".join(map(str, col)) for col in columns_as_lists.values())
    text_representation