# MedLitRAG: a RAG for scientific queries 
## The idea
MedLitRAG answers scientific queries on predefined topics of choice by using literature published on PubMed.


## Practical overview
### Knowledge base
MedLitRAG uses as base of knowledge a collection of PubMed abstracts. They are chunked, and those chunk units are embedded and stored in FAISS.

### Q&A process
After a user inputs a query a series of steps is performed:
1) The input query gets embedded into a vector
2) A fast nearest-neighbor search is performed over FAISS
3) The 20 chunks from the base knowledge collection most similar to the query are retrieved 
4) A reranker re-scores them and keeps the best 5
5) The 5 chunks are passed to the LLM alongside the input query as context
6The LLM generates an answer grounded in those specific chunks (that can be traced back as sources)

## Requirements
- Python 3.12+
- requirements.txt dependencies

## Setup
Create a `.env` file in the project root with:
```
ENTREZ_EMAIL=your@email.com
TOPICS=topic one,topic two,topic three
```
`ENTREZ_EMAIL` is required by the NCBI Entrez API. `TOPICS` is a comma-separated list of search terms of personal choice, used to build the knowledge base.

## Status
Early development.

## License
MIT
