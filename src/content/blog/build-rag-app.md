---
title: Build a Retrieval Augmented Generation (RAG) App
excerpt: RAG is a technique for augmenting LLM knowledge with additional data. The process of bringing the appropriate information and inserting it into the model prompt is known as Retrieval Augmented Generation (RAG).
publishDate: 'Jul 28 2024'
isFeatured: true
tags:
  - Learn with me
---

Today we are following https://python.langchain.com/v0.2/docs/tutorials/rag/ to build a RAG app.

## What’s RAG

RAG is a technique for augmenting LLM knowledge with additional data. The process of bringing the appropriate information and inserting it into the model prompt is known as Retrieval Augmented Generation (RAG).

A RAG has two components:

- Indexing: store the unstructured data
- Retrieval and generation: the actual RAG chain

### Indexing

- Load: load the data using Document Loader
- Split: use text splitter to break large document into smaller chunks
- Store: embedding and vector store

![Untitled](https://python.langchain.com/v0.2/assets/images/rag_indexing-8160f90a90a33253d0154659cf7d453f.png)

### Retrieval and generation

- Retrieve: Given an input, retrieve relevant splits from the vector store
- Generate: LLM to produce the output using the retrieved data

![Untitled](https://python.langchain.com/v0.2/assets/images/rag_retrieval_generation-1046a4668d6bb08786ef73c56d4f228a.png)

## Let’s code

### Load

LangChain provides 160+ types of [DocumentLoader](https://python.langchain.com/v0.2/docs/integrations/document_loaders/) to load different types of documents.

[Document loaders | 🦜️🔗 LangChain](https://python.langchain.com/v0.2/docs/integrations/document_loaders/)

In this tutorial, we are using [WebBasedLoader](https://python.langchain.com/v0.2/docs/integrations/document_loaders/web_base/) which uses [BeautifulSoup](https://beautiful-soup-4.readthedocs.io/en/latest/) to parse web content to text.

```python
import bs4
from langchain_community.document_loaders import WebBaseLoader

# Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

print(docs[0].page_content)
```

### Split

The document is too big to fit in the context of many models. We will split the document into chunks of 1000 characters with 200 characters of overlap between chunks. The overlap helps mitigate the possibility of separating a statement from important context related to it.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

len(all_splits) # 66 Documents
```

### Store

We will embed and store all documents into a vector store:

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings(model="text-embedding-3-small"))
```

LangChain supports 50+ [Embeddings](https://python.langchain.com/v0.2/docs/integrations/text_embedding/) and 60+ [Vector Stores](https://python.langchain.com/v0.2/docs/integrations/vectorstores/).

### Retrieve

```python
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6}) # Get top 6 results

retrieved_docs = retriever.invoke("What are the approaches to Task Decomposition?")

len(retrieved_docs) # 6

print(retrieved_docs[0].page_content)
```

### Generate

We will use a prompt template to make the process easier.

[LangChain hub](https://smith.langchain.com/hub) is a place full of different prompts. We will be using this [rag prompt](https://smith.langchain.com/hub/rlm/rag-prompt).

```python
from langchain import hub

prompt = hub.pull("rlm/rag-prompt")

example_messages = prompt.invoke(
    {"context": "filler context", "question": "filler question"}
).to_messages()

print(example_messages[0].content)

# You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
# Question: filler question
# Context: filler context
# Answer:
```

Let’s combine retriever, template, model together using a chain:

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

for chunk in rag_chain.stream("What is Task Decomposition?"):
    print(chunk, end="", flush=True)\
```

## Relative Reading

Read the blog to understand how to let LLM return source for a QA app:

[How to get your RAG application to return sources | 🦜️🔗 LangChain](https://python.langchain.com/v0.2/docs/how_to/qa_sources/)
