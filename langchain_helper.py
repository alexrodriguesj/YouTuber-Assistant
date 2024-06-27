#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[12]:


from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate

import openai
from dotenv import load_dotenv
import os


# # Set API KEY

# In[19]:


load_dotenv()

openai_api_key = openai.api_key = 'your-api-key-here'

embeddings = OpenAIEmbeddings(
    openai_api_key=openai_api_key,
)


# # Create Funcions

# ## Vector from youtube URL

# In[28]:


def create_vector_from_yt_url(video_url: str) -> FAISS:
    """Give me explanation of funcion
    """
    loader = YoutubeLoader.from_youtube_url(video_url, language="en")
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db


# ## Get Responses

# In[26]:


def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        openai_api_key=openai_api_key,
    )

    chat_template = ChatPromptTemplate.from_messages(
        [
            (
                "user",
                """You are an assistant that answers questions about YouTube videos based on
        their transcriptions.

        Answer the following question: {question}
        Searching in the following transcriptions: {docs}

        Use only the information from the transcription to answer the question. If you don't know, respond
        with "I don't know".

        Your answers should be detailed and verbose.
        """,
            )
        ]
    )

    chain = LLMChain(llm=llm, prompt=chat_template, output_key="answer")

    response = chain({"question": query, "docs": docs_page_content})

    return response, docs


# In[ ]:


if __name__ == "__main__":
    video_url = ""
    query = ""
    db = create_vector_from_yt_url(video_url)
    
    response, docs = get_response_from_query(
        db, query
    )
    print(response)

