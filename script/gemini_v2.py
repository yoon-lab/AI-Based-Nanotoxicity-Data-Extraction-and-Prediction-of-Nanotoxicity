# -*- coding: utf-8 -*-


# 0. Enter username (output folder will made for this name)
"""

user_name = "Eunyong_Ha"

import time
notebook_start_time = time.time()

"""# 1. Basic prompt template of LangChain enviroment"""

template = """
You are a data analyst specializing in nano-toxicity.
Your task is extracting data about the physicochemical and cytotoxic properties of nanomaterials used in the research paper.
Use the following pieces of instruction and format to answer the user's questions.
Don't be lazy.
Be very strict and answer the question very accurate and scientific manner.
If data is not specified in the paper, just answer 'None'.
Do not use a full sentence.
Context: {context}
Question: {question}
Format instruction: {format_instructions}
"""

"""# 2. Material 정보를 추출하는 프롬프트"""

q_mat = """
Just answer with following format: 'Nanomaterial name (Nanomaterial type)'. Do not use a full sentence.
If there is no value, assign 'None'.
Question: What nanoparticles were used in the characterization experiments in the research paper? Refer to the following descriptions.
Description 1. Provide the nanoparticles that were used for core size(i.e. primary size, nominal size) or hydrodynamic size(i.e. Z-average size) or surface charge(i.e. zeta potential) or surface area in the research paper.
Description 2. Provide information on all the nanoparticles actually used in the authors' experiments. Do not include nanoparticles used in other papers (i.e., mentioned in the reference) in your answer.
Description 3. If there were multiple nanoparticles of the same type, the author would have named them differently. What did the author name them?
Description 4. Write the name of the nanoparticle followed by its type in chemical formula within parentheses.
Description 5. Several examples of the format.
-Written in the document: Al2O3 NPs, ZnO, SiO2, Fe2O3 (normal form), Response: Al2O3 (Al2O3), ZnO (ZnO), SiO2 (SiO2), Fe2O3 (Fe2O3).
-Written in the document: T10, T100 (labeled differently according to 'size'), Response: T10 (TiO2), T100 (TiO2).
-Written in the document: ZnAc, ZnChl (ZnO-Acetate, ZnO-Chloride; labeled differently according to 'chemical'), Response: ZnAc (ZnO), ZnChl (ZnO).
-Written in the document: TiO2-PVP, TiO2-Citrate (labeled differently according to 'coating'), Response: TiO2-PVP (TiO2), TiO2-Citrate (TiO2).
-Written in the document: P25, Nanofilament (labeled differently according to 'manufacturer'), Response: P25 (TiO2), Nanofilament (TiO2).
-Written in the document: CuO-USA, CuO-UK (labeled differently according to 'location'), Response: CuO-USA (CuO), CuO-UK (CuO).
Description 6. Do not omit the information.
Description 7. Do not write 'NP' or 'nanoparticles' followed by nanoparticles'name
"""

"""# 3. PChem/Tox 정보를 추출하는 프롬프트
- Pchem
- Tox
"""

import pprint
from typing import Any, Dict
import datetime
from pytz import timezone

import pandas as pd
from langchain.output_parsers import PydanticOutputParser
#from langchain.output_parsers import PandasDataFrameParser

from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator

class pchem_product_info(BaseModel):
    mat_name: str = Field(description="Question: What nanoparticles were used in the characterization experiments in the research paper? Refer to the following descriptions. Description 1. Provide the nanoparticles that were used for core size(i.e. primary size, nominal size) or hydrodynamic size(i.e. Z-average size) or surface charge(i.e. zeta potential) or surface area in the research paper. Description 2. Provide information on all the nanoparticles actually used in the authors' experiments. Do not include nanoparticles used in other papers (i.e., mentioned in the reference) in your answer. Description 3. If there were multiple nanoparticles of the same type, the author would have named them differently. What did the author name them? Description 4. Write the name of the nanoparticle followed by its type in chemical formula within parentheses.   Description 5. Several examples of the format. -Written in the document: Al2O3 NPs, ZnO, SiO2, Fe2O3 (normal form), Response: Al2O3 (Al2O3), ZnO (ZnO), SiO2 (SiO2), Fe2O3 (Fe2O3). -Written in the document: T10, T100 (labeled differently according to 'size'), Response: T10 (TiO2), T100 (TiO2). -Written in the document: ZnAc, ZnChl (ZnO-Acetate, ZnO-Chloride; labeled differently according to 'chemical'), Response: ZnAc (ZnO), ZnChl (ZnO). -Written in the document: TiO2-PVP, TiO2-Citrate (labeled differently according to 'coating'), Response: TiO2-PVP (TiO2), TiO2-Citrate (TiO2). -Written in the document: P25, Nanofilament (labeled differently according to 'manufacturer'), Response: P25 (TiO2), Nanofilament (TiO2). -Written in the document: CuO-USA, CuO-UK (labeled differently according to 'location'), Response: CuO-USA (CuO), CuO-UK (CuO). Description 6. Do not omit the information. Description 7. Do not write 'NP' or 'nanoparticles' followed by nanoparticles'name")
    def to_dict(self):
        return {"mat_name": self.mat_name}

class pchem_mat_synthesis(BaseModel):
    mat_synthesis: str = Field(description="The following nanoparticles were synthesized by researcher? or commercially available? If nanoparticles were synthesized, just answer 'Synthesized'. If nanoparticles were commercially available, just answer 'Commercially available (with cat# or product # in parentheses).'")
    def to_dict(self):
        return {"mat_synthesis": self.mat_synthesis}

class pchem_core_size(BaseModel):
    mat_core_size: str = Field(description="What is the value of core size or core size distribution (i.e. primary size, nominal size) of each material? Refer to the following format to answer. 0. TEM, SEM, AFM size. 1. Do not include unit. 2. Do not use a full sentence. 3. If there is no value, assign 'None'. 4. Do not include calculated size. 5. If the values are represented as a range, they are represented in the following format: value1-value2. ex) 50-100. 6. If the values are represented with an error rate '±', they are represented in the following format: value±error rate. ex) 35±10.")
    def to_dict(self):
        return {"mat_core_size": self.mat_core_size}

class pchem_hydrodynamic_size(BaseModel):
    mat_hydrodynamic_size: str = Field(description="What is the value of hydrodynamic size (i.e., Z-average size, size in media) of each material? Please provide details on the sizes under various conditions or in different media. 0. DLS size. 1. Do not include unit. 2. Do not include an explanation about hydrodynamic size. Just give me the value. 3. Do not use a full sentence. 4. If there is no value, assign 'None'. 5. If multiple values exist for each material, divide the value using ';', add parentheses after the value, and write the conditions in the parentheses. 6. Please refer to the following format when you write down the conditions in parentheses. Format: [Classification: detailed conditions]. ex) 50 (Solvent: water); 100 (Solvent: medium), 30 (Time: 2 h); 50 (Time: 24 h). Kind of Classification: Solvent, Time, Concentration, pH. 7. If the values are represented as a range, they are represented in the following format: value1-value2. ex) 50-100. 8. If the values are represented with an error rate '±', they are represented in the following format: value±error rate. ex) 35±10.")
    def to_dict(self):
        return {"mat_hydrodynamic_size": self.mat_hydrodynamic_size}

class pchem_surface_charge(BaseModel):
    mat_surface_charge: str = Field(description="What is the value of surface charge (i.e., Zeta potential) of each material? Please provide details on the sizes under various conditions or in different media. 1. Do not include unit. 2. Do not use a full sentence. 3. If there is no value, assign 'None'. 4. If multiple values exist for each material, divide the value using ';', add parentheses after the value, and write the conditions in the parentheses. 5. Please refer to the following format when you write down the conditions in parentheses. Format: [Classification: detailed conditions]. ex) (negative)10 (Solvent: water); (positive)21 (Solvent: medium), (positive)30 (Time: 2 h); 50 (Time: 24 h). Kind of Classification: Solvent, Time, Concentration, pH. 6. If the values are represented as a range, they are represented in the following format: ‘value1’to’value2’. ex) 50to100. 7. If the values are represented with an error rate '±', they are represented in the following format: value±error rate. ex) 35±10. 8. If you encounter no number, assign 'None' (e.g. (just) - dash format).")
    def to_dict(self):
        return {"mat_surface_charge": self.mat_surface_charge}

class pchem_surface_area(BaseModel):
    mat_surface_area: str = Field(description="What is the value of surface area of each material? Refer to the following format to answer. 1. Do not include unit. 2. Do not use a full sentence. 3. If there is no value, assign 'None'. 4. Do not include calculated size. 5. If the values are represented as a range, they are represented in the following format: value1-value2. ex) 50-100. 6. If the values are represented with an error rate '±', they are represented in the following format: value±error rate. ex) 35±10.")
    def to_dict(self):
        return {"mat_surface_area": self.mat_surface_area}

class tox_info(BaseModel):
    cell_type: str = Field(description="What cell lines were used in the cell viability assay? 1. Do not use a full sentence. 2. Please provide the abbreviation form of the cell name. ex) A549, THP-1, MRC-5, EA.hy926, BEAS-2B, HaCaT, L929, U87, etc. 3. If multiple cell lines were used, divide the value using ';'. ex) A549; THP-1; MRC-5.")
    cell_species: str = Field(description="What species the cell line originated from? 1. Do not use a full sentence. 2. Please refer to the following form. ex) Human, Rabbit, Mouse, Pig, etc. 3. If multiple cell lines were used and their species are different,  divide the value using ';' and write the name of the cell species followed by its cell type within parentheses. ex) Human (A549); Mouse (L929).")
    cell_organ: str = Field(description="What organ the cell line originated from? 1. Do not use a full sentence. 2. Please refer to the following form. ex) Lung, Breast, Kidney, Brain, Liver, Bronchial tube, Prostate, Spleen, etc. 3. If multiple cell lines were used and their organs are different, divide the value using ';' and write the name of the cell species followed by its cell type within parentheses. ex) Lung (A549); Fibroblast (L929).")
    cell_assay: str = Field(description="Which cell viability assays or cytotoxicity assay were conducted in this paper? 1. Do not use a full sentence. 2. Only reference the following cell viability assays, and if none, assign as 'None'. 2. Please refer to the following form. ex) CCK-8, MTT, MTS, WST, Alamar blue, CellTiter-Glo, Neutral Red, NRU, Trypan blue, XTT, Calcein-AM, BrdU, EdU, Propidium iodide, Hoechst33342 assays. 3. If multiple cell viability assays were used, divide the value using ';'. ex) MTT; MTS; CCK-8.")
    cell_classification: str = Field(description="Please determine the cell type: whether it is a normal cell or a cancer cell? 1. Do not use a full sentence. 2. Just answer, 'Normal' or 'Cancer'.")
    def to_dict(self):
        return {"cell_type": self.cell_type, "cell_species": self.cell_species, "cell_organ": self.cell_organ, "cell_assay": self.cell_assay,"cell_classification": self.cell_classification}

"""# 4. Zotereo와 LangChain연결
- Zotero collection 설정
- PDF 파일 폴더와 output 폴더 설정
- Zotero에서 데이터 끌고 오기
- Get PDF file list and IDs
"""

import os
from os.path import join, basename, splitext
import subprocess
from glob import glob
from shutil import copy
from random import shuffle, seed

from pyzotero import zotero

zot = zotero.Zotero(library_id, 'library_type', 'zotero_api_key') ## fill zotero.Zoter(library_id, library_type, zotero_api_key)

collections = {c['data']['name']: c for c in zot.collections()}

collection_names = []
for key, value in collections.items():
    #print(key)
    collection_names.append(key)

collection_names

selected_collection = "enter_collection_name" # 위의 리스트에서 collection 이름을 복사해 넣음.
pdf_folder = "/home/pdf_" + selected_collection.replace(" ", "_")
print("PDF 파일들은 " + pdf_folder + "에 저장됩니다.")

import os

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created at {folder_path}")
    else:
        print(f"Folder already exists at {folder_path}")

output_folder = "output_" + user_name
create_folder(output_folder)
create_folder(pdf_folder)

import os
from os.path import join, basename, splitext
import subprocess
from glob import glob
from shutil import copy
from random import shuffle, seed

from pyzotero import zotero
zot = zotero.Zotero(library_id, 'library_type', 'zotero_api_key') ## fill zotero.Zoter(library_id, library_type, zotero_api_key)

collections = {c['data']['name']: c for c in zot.collections()}
collection = collections[selected_collection]
key = collection['key']
items = [d for d in zot.everything(zot.collection_items(key))]

pdf_dict = {}
for item in items:
    pdf_dict[item['data']['key']] = item['data']['title']

items = [d for d in zot.everything(zot.collection_items(key))]

for item in items:
    children = [c for c in zot.children(item['key'])]

    pdfs = [c for c in children if c['data'].get('contentType') == 'application/pdf']
    #print(pdfs)

    if not children:
        print('\nMissing documents {}\n'.format(item['data']['title']))
    elif not pdfs:
        print('\nNo PDFs {}\n'.format(item['data']['title']))
    elif len(pdfs) != 1:
        print('\nToo many PDFs {}\n'.format(item['data']['title']))
    else:
        doc = pdfs[0]
        print(doc['data']['filename'])
        pdf_file_path = os.path.join(pdf_folder, '{}.pdf'.format(doc['key']))
        if not os.path.exists(pdf_file_path):
            zot.dump(doc['key'], '{}.pdf'.format(doc['key']), pdf_folder)
            print(f"{pdf_file_path} is downloaded.")
        else:
            print(f"{pdf_file_path} already exists")

pdf_files = []

for file_path in os.listdir(pdf_folder):
    if os.path.isfile(os.path.join(pdf_folder, file_path)):
        pdf_files.append(os.path.join(pdf_folder, file_path))

print(pdf_files)

# get pdf ids and store as pdf_ids
pdf_ids = []
for file_path in pdf_files:
    # get base name of file_pafth and remove .pdf
    base_name = os.path.basename(file_path)
    base_name = os.path.splitext(base_name)[0]
    pdf_ids.append(base_name)
print(pdf_ids)

"""# 5. Gemini AI with LangChain

1. 환경 변수 'GOOGLE_API_KEY'에 고정된 API key를 설정. 만약 환경변수가 설정되어있지 않다면, 사용자에게 api key 요구
"""

import os
import getpass
os.environ["GOOGLE_API_KEY"]="google_API_Key"
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Provide your Google API Key")

"""2.Gemini LLM 모델 및 Embedding 모델 설정 + 기존 코드와 변경된 점

from langchain_google_genai import ChatGoogleGenerativeAI #LangChain-Gemini를 chain시켜주는 코드.
from langchain_google_genai import GoogleGenerativeAIEmbeddings #Google에서 제공하는 embedding model.

LLM 모델과 Embedding모델 list 코드:
"""

from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFium2Loader



embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


from langchain_openai import OpenAIEmbeddings

"""3. 파일 로딩 후 text splitting and FAISS indexing
- PyPDFium2Loader = pdf 업로드
- textsplitter
- FAISS index 붙이기
"""

def get_pdf_text(file_pafth):
    # get base name of file_pafth and remove .pdf
    base_name = os.path.basename(file_pafth)
    base_name = os.path.splitext(base_name)[0]
    print(base_name)

    em_path = "/home/workspace/embed/gemini_em3/" + base_name + "_gemini"
    if not os.path.exists(em_path):
        #load = PdfReader(file_pafth)
        #load = PyPDFLoader(file_pafth)
        load = PyPDFium2Loader(file_pafth)
        document = load.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        texts = text_splitter.split_documents(document)
        db = FAISS.from_documents(texts, embeddings)

        # make file name as base_name_faiss and save it
        db.save_local(em_path)
        print(f"{em_path} is generated.")
    else:
        print(f"{em_path} already exists")

# run get_pdf_text for all pdf_files
for file_path in pdf_files:
    get_pdf_text(file_path)

"""4. LLM import.

"""

from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

chat = ChatGoogleGenerativeAI(model="gemini-1.5-pro",
                             temperature=0)

from langchain.callbacks import get_openai_callback

openai_cb = {}

def get_answer(doc_id, q, parser):
    db = FAISS.load_local("/home/workspace/embed/gemini_em3/" + doc_id + "_gemini", embeddings,allow_dangerous_deserialization=True)
    prompt = PromptTemplate(
        template=template,
        input_variables=['context', 'question'],
        partial_variables = {"format_instructions": parser.get_format_instructions()},)
    retriever = db.as_retriever(search_kwargs={'k': 10})
    qa_llm = RetrievalQA.from_chain_type(llm = chat,
                                        chain_type = 'stuff',
                                        retriever = retriever,
                                        return_source_documents = True,
                                        chain_type_kwargs = {'prompt': prompt})
    with get_openai_callback() as cb:
        output = qa_llm({'query': q})
        now = datetime.datetime.now(timezone('Asia/Seoul'))
        now = now.strftime("%Y%m%d_%H%M%S")
        openai_cb[now] = cb
    return output

output_parser = CommaSeparatedListOutputParser()
mats_all_paper = {pdf_id: get_answer(pdf_id, q_mat, output_parser)['result'] for pdf_id in pdf_ids}

"""# 6. 정보 추출


"""

mats_all_paper

# OutputFixingPaser

from langchain.output_parsers import OutputFixingParser
fixing_parser = OutputFixingParser.from_llm(parser = output_parser, llm = chat)

mats_all_paper_fix = {}
for key, value in mats_all_paper.items():
    value_fix = fixing_parser.parse(value)
    mats_all_paper_fix[key] = value_fix

mats_all_paper_fix

from dataclasses import dataclass

@dataclass
class gpt_responses:
    pdf_id: str
    mat_name: str
    pchem_product_info: str
    pchem_mat_synthesis: str
    pchem_core_size: str
    pchem_hydrodynamic_size: str
    pchem_surface_charge: str
    pchem_surface_area: str

"""import json
import textwrap
import langchain_core.output_parsers


def get_sub_answers(mat, q, key, data_class):
    schema = data_class.schema()
    field_names = list(schema['properties'].keys())
    field_names.insert(0, 'key')
    #print(field_names)

    output_parser = PydanticOutputParser(pydantic_object=data_class)
    ans = get_answer(key, q, output_parser)

    fixing_parser = OutputFixingParser.from_llm(parser = output_parser, llm = chat)
    text = ans['result']
    py = fixing_parser.parse(text)
    py_dict = py.to_dict()

    #print(py_dict)

    pchem_df = pd.DataFrame.from_records(py_dict, index = ["0"])
    pchem_df.insert(0, 'key', key)
    #pchem_df.insert(0, 'ref', pdf_dict[key])
    print(pchem_df)

    return(pchem_df)

#i = "TiO2 P25–70 nano-TiO2"
#q = "please pull out material information of " + i + " in the document."
#tmp = get_sub_answers(i, q, "95ASP9SZ", pchem_product_info)
"""

import json
import pandas as pd
import langchain_core.output_parsers

def get_sub_answers(mat, q, key, data_class):
    schema = data_class.schema()
    field_names = list(schema['properties'].keys())
    field_names.insert(0, 'key')

    output_parser = PydanticOutputParser(pydantic_object=data_class)
    ans = get_answer(key, q, output_parser)

    fixing_parser = OutputFixingParser.from_llm(parser=output_parser, llm=chat)

    text = ans['result']
    print(text)

    try:
        py = fixing_parser.parse(text)
        py_dict = py.to_dict()

        pchem_df = pd.DataFrame.from_records([py_dict])  # 여기에서 []를 추가하여 하나의 레코드로 만듭니다.
        pchem_df.insert(0, 'key', key)
        print(pchem_df)

        return pchem_df
    except Exception as e:
        print(f"Error parsing response for key {key}: {e}")
        return None

# 예시 호출
# i = "TiO2 P25–70 nano-TiO2"
# q = "please pull out material information of " + i + " in the document."
# tmp = get_sub_answers(i, q, "95ASP9SZ", pchem_product_info)

#####아래 코드 지우기@@@@

"""all_pchem_dfs = []
all_gpt_responses = []

for key, value in mats_all_paper_fix.items():

    # make for loop to print each value
    for i in value:
        print(key + ": " + i)

        q = "please pull out material information of " + i + " in the document."
        pchem_df = get_sub_answers(i, q, key, pchem_product_info)

         #q = "please pull out surface charge information of the material in the document."
        q = "please pull out surface charge information of " + i + " in the document."
        pchem_df_add = get_sub_answers(i, q, key, pchem_surface_charge)
        pchem_df = pd.merge(pchem_df, pchem_df_add, on='key')

        pchem_df.insert(0, 'ref', pdf_dict[key])

        #gpt_res = gpt_responses(key, i, ans_info, ans_core, ans_hydro, ans_s_charge, ans_s_area)
        #all_gpt_responses.append(gpt_res)

        # Append pchem_df to the list
        all_pchem_dfs.append(pchem_df)

# Combine all pchem_df DataFrames into a single DataFrame
all_pchem_df = pd.concat(all_pchem_dfs, ignore_index=True)
all_pchem_df['ref'] = all_pchem_df['key'].map(lambda x: pdf_dict[x] if x in pdf_dict else None)
"""

all_pchem_dfs = []
all_gpt_responses = []

for key, value in mats_all_paper_fix.items():

    # make for loop to print each value
    for i in value:
        print(key + ": " + i)

        q = "please pull out material information of " + i + " in the document."
        pchem_df = get_sub_answers(i, q, key, pchem_product_info)

        q = "please pull out material synthesis information of " + i + " in the document."
        pchem_df_add = get_sub_answers(i, q, key, pchem_mat_synthesis)
        pchem_df = pd.merge(pchem_df, pchem_df_add, on='key')

        #q = "please pull out core size information of the material in the document."
        q = "please pull out core size information of " + i + " in the document."
        pchem_df_add = get_sub_answers(i, q, key, pchem_core_size)
        pchem_df = pd.merge(pchem_df, pchem_df_add, on='key')

        #q = "please pull out hydrodynamic size information of the material in the document."
        q = "please pull out hydrodynamic size information of " + i + " in the document."
        pchem_df_add = get_sub_answers(i, q, key, pchem_hydrodynamic_size)
        pchem_df = pd.merge(pchem_df, pchem_df_add, on='key')

        #q = "please pull out surface charge information of the material in the document."
        q = "please pull out surface charge information of " + i + " in the document."
        pchem_df_add = get_sub_answers(i, q, key, pchem_surface_charge)
        pchem_df = pd.merge(pchem_df, pchem_df_add, on='key')

        #q = "please pull out surface area information of the material in the document."
        q = "please pull out surface area information of " + i + " in the document."
        pchem_df_add = get_sub_answers(i, q, key, pchem_surface_area)
        pchem_df = pd.merge(pchem_df, pchem_df_add, on='key')
        pchem_df.insert(0, 'ref', pdf_dict[key])

        #gpt_res = gpt_responses(key, i, ans_info, ans_core, ans_hydro, ans_s_charge, ans_s_area)
        #all_gpt_responses.append(gpt_res)

        # Append pchem_df to the list
        all_pchem_dfs.append(pchem_df)

# Combine all pchem_df DataFrames into a single DataFrame
all_pchem_df = pd.concat(all_pchem_dfs, ignore_index=True)
all_pchem_df['ref'] = all_pchem_df['key'].map(lambda x: pdf_dict[x] if x in pdf_dict else None)
###

"""# Pchem 정보 추출 결과"""

all_pchem_df

"""# Pchem 정보 추출 결과 저장"""

import datetime
from pytz import timezone

now = datetime.datetime.now(timezone('Asia/Seoul'))
now = now.strftime("%Y%m%d_%H%M%S")
o = os.path.join(output_folder, "pchem_gtp_output_" + now + ".xlsx")
all_pchem_df.to_excel(o)
print(o + "로 저장되었습니다.")

"""# Tox 정보 추출"""

all_tox_dfs = []
all_gpt_responses = []

for key, value in mats_all_paper_fix.items():

    # make for loop to print each value
    for i in value:
        print(key)

        q = "please pull out cytotoxicity information in the document."
        tox_df = get_sub_answers(i, q, key, tox_info)

        #gpt_res = gpt_responses(key, i, ans_info, ans_core, ans_hydro, ans_s_charge, ans_s_area)
        #all_gpt_responses.append(gpt_res)

        # Append pchem_df to the list
        all_tox_dfs.append(tox_df)

# Combine all pchem_df DataFrames into a single DataFrame
all_tox_df = pd.concat(all_tox_dfs, ignore_index=True)
all_tox_df['ref'] = all_tox_df['key'].map(lambda x: pdf_dict[x] if x in pdf_dict else None)

all_tox_df

"""# Tox 정보 추출 결과"""

import datetime
from pytz import timezone

now = datetime.datetime.now(timezone('Asia/Seoul'))
now = now.strftime("%Y%m%d_%H%M%S")
o = os.path.join(output_folder, "tox_gtp_output_" + now + ".xlsx")
all_tox_df.to_excel(o)
print(o + "로 저장되었습니다.")

# 마지막 셀에 추가
notebook_end_time = time.time()
print(f"노트북 전체 실행 시간: {notebook_end_time - notebook_start_time}초")

"""# Tokens and cost"""

cost_list = []
for key, value in openai_cb.items():
    #print(value.total_cost)
    cost_dict = {}
    cost_dict["Date Time"] = key
    cost_dict["Total tokens"] = value.total_tokens
    cost_dict["Total cost ($)"] = value.total_cost
    cost_list.append(cost_dict)

cost_df = pd.DataFrame(cost_list)
sums = cost_df.select_dtypes(include='number').sum()
sums_dict = {"Date Time": "Total", "Total tokens": sums["Total tokens"], "Total cost ($)": sums["Total cost ($)"]}

cost_df = pd.concat([cost_df, pd.DataFrame([sums_dict])])
cost_df

import datetime
from pytz import timezone

now = datetime.datetime.now(timezone('Asia/Seoul'))
now = now.strftime("%Y%m%d_%H%M%S")
o = os.path.join(output_folder, "token_and_cost_" + now + ".xlsx")
cost_df.to_excel(o)
print(o + "로 저장되었습니다.")

# 마지막 셀에 추가
notebook_end_time = time.time()
print(f"노트북 전체 실행 시간: {notebook_end_time - notebook_start_time}초")

