from rapidocr import RapidOCR
from PIL import Image
import io
from groq import Groq
import base64
import time
import os
import pymupdf
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.documents import Document

from typing import TypedDict,Annotated,Union,List, Literal
from pydantic import BaseModel, Field
import PIL
from langchain_community.retrievers import ArxivRetriever
from langgraph.types import Command, interrupt
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
import arxiv
from sentence_transformers import CrossEncoder

log_messages = []
api_key = "YOUR_API_KEY"

class Parser:
    def __init__(self,api_key:str):
        self._api_key = api_key
        
    def _extract_captions_of_images(self,doc,page):
        imgs = page.get_images()
        client = Groq(api_key=self._api_key)
        captions = {}

        for i in range(len(imgs)):
            print("...VLM Called...",end="")

            xref = imgs[i][0]
            base_image = doc.extract_image(xref)

            #if image is unicolor that means it is either mask or artifact
            if base_image['colorspace']==1:
                continue

            image_bytes = base_image["image"]

            image_ext = base_image["ext"]

            image = Image.open(io.BytesIO(image_bytes))
            image = image.resize((360,180))
            output = io.BytesIO()
            # image
            image.save(output, format=image_ext)
            base64_image = base64.b64encode(output.getvalue()).decode('utf-8')

            start_time = time.time()
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe the image in 50 words as much as possible/"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ],
                model="meta-llama/llama-4-scout-17b-16e-instruct",
            )

            captions[f"img_{xref}"] = chat_completion.choices[0].message.content

        return captions
        
    def _extract_using_ocr(self,page):
        print("...OCR called...",end="")
        img = page.get_pixmap()
        img_bytes = img.tobytes()
        image = Image.open(io.BytesIO(image_bytes))

        if image.width > image.height:
            image = image.rotate(90,expand=True)

        image = image.resize((400,800))
        result = engine(image)
        text = "\n".join(txt for txt in result.txts)
        
        return text

    def _extract_text_excluding_tables(self,page):
        tables = page.find_tables(strategy="lines_strict")
        table_bboxes = [table.bbox for table in tables]

        def is_inside_any_table_bbox(bbox):
            for table_bbox in table_bboxes:
                # print(table_bbox)
                if pymupdf.Rect(table_bbox).intersects(pymupdf.Rect(bbox)):
                    return True
            return False

        # Get all text blocks
        blocks = page.get_text("blocks")  
        filtered_text = [
            block[4] for block in blocks
            if not is_inside_any_table_bbox(block[:4])
        ]

        return "\n".join(filtered_text)
    def _extract_table_content(self,page):
        tables = page.find_tables()
        tables_list = [table.to_markdown() for table in tables]

        text = "\n".join(text for text in tables_list)

        return text

    def parse_pdf(self,path):
        global log_messages
        log_messages.append("Parsing the pdf")
        doc = pymupdf.open(path)
        parsed = []

        for i in range(doc.page_count):
            print(f"Page {i+1}",end="")

            full_pg = {}
            start_time = time.time()
            pg = doc.load_page(i)

            text = self._extract_text_excluding_tables(pg)

            if text == "" or text == []:
                text = self._extract_using_ocr(pg)
                img = ""
                table = ""
            else:
                img = self._extract_captions_of_images(doc,pg)
                table = self._extract_table_content(pg)

            full_pg['text'] = text
            full_pg['tables'] = table
            full_pg['imgs'] = img
            full_pg['page'] = i+1

            parsed.append(full_pg)
            print(f"..Done.. {time.time()-start_time}")

        log_messages.append("PDF parsed")
        return parsed

class AgentState(TypedDict):
    query: str
    pdf_path: str
    result: str
    imgs: list[str]
    paper_url: str
    next_node: str
    prev_node: str

class Classifier(BaseModel):
    classify_intent: Literal["fetch_paper","RAG_qna"] = Field(
        description = """Classify the intent of user query:
        'fetch_paper' if user is asking for specific research paper for this the query should have words like 'retrieve','fetch',etc.
        'RAG_qna' if user is asking questions directly."""
    )

class Agent:
    def __init__(self,vdb_name:str,vdb_path:str,api_key:str):
        self._api_key = api_key
        self._embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
        self._cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self._vectorstore = Chroma(
            collection_name=vdb_name,
            embedding_function=self._embedding,
            persist_directory=vdb_path
        )
        self._retriever = self._vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={'k': 2}
            )
        self._parser = Parser(api_key=api_key)
        self._text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=500, chunk_overlap=100
        )
        self._llm = init_chat_model('groq:llama-3.1-8b-instant',temperature=0.2,api_key= api_key)
        self._llm_router = self._llm.with_structured_output(Classifier)
        self._retriever_arxiv = ArxivRetriever(
            load_max_docs=1,
            get_ful_documents=True,
        )
        
    def _parse_and_embed(self,state:AgentState):
        global log_messages
        log_messages.append("In parse and embed")
        
        print("In parse_and_embed")
        
        result = self._parser.parse_pdf(state['pdf_path'])
        
        docs_list = [Document(page_content=page['text']+"\n\n"+page['tables']+"\n\n"+"\n".join(page['imgs'][key] for key in page['imgs'].keys()),
                          metadata={"page": page['page'],"imgs":False if not page['imgs'] else ",".join(img.split('_')[1] for img in page['imgs']), 
                                   'pdf_path':state['pdf_path']}) for page in result]

        doc_splits = self._text_splitter.split_documents(docs_list)

        self._vectorstore.add_documents(documents=doc_splits)
        
        print("parsed and embedded the pdf")

        if state['prev_node'] == 'user_confirmation':
            return {'result':"Parsed the pdf! now you can ask questions related to it.",'next_node':'END','prev_node':'parse_and_embed'}
        else:
            return {'next_node':"classify_query_intent",'prev_node':'parse_and_embed'}

    def _fetch_arxiv_paper(self,state:AgentState):
        global log_messages
        log_messages.append("In fetch_arxiv_paper")
        
        print("In fetch_arxiv_paper")
        all_papers=[]

        prompt = ChatPromptTemplate.from_messages([
            ("system","""You are a query analyser and reformatter for research paper searching. You can shorten the query and only include relevant words for efficient search on research paper websites.Only include the query in output nothing else. """),
            ("human","Input:{input}")
        ])
        chain = prompt | self._llm

        result = chain.invoke({
            "input":state['query']
        })

        docs = self._retriever_arxiv.invoke(result.content)
    #     print(docs)

        for doc in docs:
            paper_info = {
                "title": doc.metadata['Title'],
                "summary": doc.page_content,
                "url": doc.metadata['Entry ID']
            }
            all_papers.append(paper_info)

        text = f"The relevant paper found:\n Title: {all_papers[0]['title']} \n\n Summary: {all_papers[0]['summary']}\n\n Source: {all_papers[0]['url']}"

        return {'result':text,'paper_url': all_papers[0]['url'],'next_node':'user_confirmation','prev_node':'fetch_arxiv_paper'}

    def _rag_and_generate(self,state:AgentState):
        global log_messages
        log_messages.append("In rag_and_generate")
        
        print("In rag_and_generate")

        #query expander
        qe_prompt = ChatPromptTemplate.from_messages([
            ("system","""You are a query expander agent.You expand the query for better retrieval from vector database.Your task is to create a expected answer .Just give the expected made up answer to the query, Nothing else.Maximum 50 words."""),
            ("human","Input:{input}")
        ])
        qe_chain = qe_prompt | self._llm
        result = qe_chain.invoke({
            "input":state['query']
        })
    #     print(f"eq: {result.content}")

        docs1 = self._retriever.invoke(result.content)
        docs2 = self._retriever.invoke(state['query'])

        docs = []
        for doc in docs1:
            if not doc in docs:
                docs.append(doc)
        for doc in docs2:
            if not doc in docs:
                docs.append(doc)

        pairs = [[state['query']+"\n\n"+result.content, doc.page_content] for doc in docs]

        #reranker
        scores = self._cross_encoder.predict(pairs)

        comb_text = sorted(dict(zip(scores,list(range(0,len(pairs))))).items(),reverse=True)
        top_3_content = "\n\n".join(docs[t[1]].page_content for t in comb_text[:3])


        prompt = ChatPromptTemplate.from_messages([
            ("system","""Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer"""),
            ("human","Context:{context}\n\nInput:{input}")
        ])
        chain = prompt | self._llm

        result = chain.invoke({
            "input":state['query'],
            "context":top_3_content
        })

        try:
            pdf = pymupdf.open(docs[comb_text[0][1]].metadata['pdf_path'])
        except Exception as e:
            print(f"ERROR:{e}")
            print(docs[comb_text[0][1]].metadata['pdf_path'])
            return
        imgs = []
        for t in comb_text:
            if not docs[t[1]].metadata['imgs']:
                continue

            for img_no in docs[t[1]].metadata['imgs'].split(','):
                img = pdf.extract_image(int(img_no))
                base_img = img['image']
                img_ext = img['ext']
                image = base_img
                imgs.append(image)

        return {'result': result.content,'imgs':imgs,'prev_node':"rag_and_generate"}    

    def _classify_query_intent(self,state:AgentState):
        global log_messages
        log_messages.append("In classify_query_intent")
        
        print("In classify_query_intent")
        
        result = self._llm_router.invoke(
            [
                {"role": "system", "content": "you are an excellent classifier of user query."},
                {"role": "user", "content": state['query']}
            ]
        )
        if result.classify_intent == 'fetch_paper':
            return {'next_node':'fetch_arxiv_paper','prev_node':"classify_query_intent"}
        else:
            return {'next_node':'rag_and_generate','prev_node':"classify_query_intent"}
       
    def _router(self,state:AgentState):
        print("In router")
        
        if state['pdf_path'] == None:
            return {'next_node':'classify_query_intent','prev_node':'router'}
        else:
            return {'next_node':'parse_and_embed','prev_node':'router'}
    
    def _user_confirmation(self,state:AgentState):
        print("In user_confirmation")
        human_response = interrupt({"query": "Do you want me to fetch this paper, Yes or No?"})

        if human_response in ['Yes','YES','yes']:
            arxiv_id = state['paper_url'].split('/')[-1]
            paper = next(arxiv.Client().results(arxiv.Search(id_list=[arxiv_id])))
            paper.download_pdf(dirpath="./uploads", filename=f"{arxiv_id}.pdf")
            
            return {'next_node':'parse_and_embed','pdf_path':f"./uploads/{arxiv_id}.pdf",'prev_node':'user_confirmation'}
        else:
            return {'next_node':"END",'result':"Ok! You can ask me to fetch some other paper or ask questions on fetched papers.",'prev_node':'user_confirmation'}

    def create_agent(self):
        graph = StateGraph(AgentState)
        checkpointer = MemorySaver()

        graph.add_node("Router",self._router)
        graph.add_node("parse_and_embed",self._parse_and_embed)
        graph.add_node("rag_and_generate",self._rag_and_generate)
        graph.add_node("classify_query_intent",self._classify_query_intent)
        graph.add_node("fetch_arxiv_paper",self._fetch_arxiv_paper)
        graph.add_node("user_confirmation",self._user_confirmation)

        graph.set_entry_point("Router")


        graph.add_edge("fetch_arxiv_paper","user_confirmation")
        graph.add_edge("rag_and_generate",END)

        graph.add_conditional_edges(
            "Router",
            lambda state: state['next_node'],
            {
                "parse_and_embed":"parse_and_embed",
                "classify_query_intent":"classify_query_intent",
            }
        )

        graph.add_conditional_edges(
            "classify_query_intent",
            lambda state: state['next_node'],
            {
                "rag_and_generate":"rag_and_generate",
                "fetch_arxiv_paper":"fetch_arxiv_paper",
            }
        )

        graph.add_conditional_edges(
            "user_confirmation",
            lambda state: state['next_node'],
            {
                "parse_and_embed":"parse_and_embed",
                "END":END
            }
        )
        graph.add_conditional_edges(
            "parse_and_embed",
            lambda state: state['next_node'],
            {
                "rag_and_generate":"rag_and_generate",
                "classify_query_intent":"classify_query_intent",
                "END":END
            }
        )
        return graph.compile(checkpointer=checkpointer)

agent=Agent(vdb_name="document_qna",vdb_path="./document_qna",api_key=api_key)

agent = agent.create_agent()
