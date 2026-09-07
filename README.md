# admission-rag

Local Streamlit RAG assistant for **Sanjivani University, Kopargaon, Maharashtra – 423601**. It does not contain or answer from Sanjivani College of Engineering data.

## What it does

- Automatically indexes the bundled `documents/sanjivani_2026_27_admission_data.txt`.
- Accepts additional PDF and TXT admission documents from the sidebar.
- Extracts PDF page numbers with PyMuPDF.
- Embeds chunks with `sentence-transformers/all-MiniLM-L6-v2`.
- Stores document text and metadata in persistent ChromaDB at `./chroma_db`.
- Retrieves at most four chunks using cosine distance and a configurable threshold.
- Sends only retrieved context to the local Ollama chat API.
- Shows source filename, page, academic year, and data type for each answer.
- Returns `I couldn't find this information in the available admission documents.` when no chunk passes the threshold.

## Data handling

The bundled 2026–27 values are explicitly labeled **provided project data**, not official university documents. The 1 July 2026 application start date is labeled **estimated project data** and the answer must include a verification note. Documents named or uploaded as official Sanjivani University material retain their own academic year, such as `2025-26`; old academic-year values are not silently converted to 2026–27.

The bundled data includes the B.Tech CSE fee, application dates, 160 seats, no fixed cutoff specified, no scholarship listed, and basic/AC hostel fees. It does not invent eligibility, required documents, courses, or admission procedures; those answers require retrieved university document context.

## Installation

From the `admission-rag` folder:

```bash
pip install -r requirements.txt
```

Install and start Ollama, then download the default model:

```bash
ollama pull llama3.2
```

## Start

```bash
streamlit run app.py
```

Open http://localhost:8501/.

## Use

1. Start the app. The bundled Sanjivani project data indexes automatically.
2. Upload a Sanjivani University PDF or TXT document if more knowledge is needed.
3. Click **Index Document**.
4. Ask a suggested or custom question and inspect the displayed sources.

Useful questions:

- What is the B.Tech CSE fee for 2026-27?
- When does admission start?
- What is the last date for admission?
- How many seats are available?
- Is there a cutoff?
- Is there a scholarship?
- What is the hostel fee?
- What documents are required?
- What is the eligibility for B.Tech CSE?
- What courses are available?

## Troubleshooting

- **Ollama unavailable:** Start Ollama and run `ollama pull llama3.2`. The app uses `http://localhost:11434/api/chat` by default.
- **Model unavailable:** Pull the selected model or change the model in the sidebar.
- **No answer:** Make sure a relevant Sanjivani University document is indexed. The app intentionally does not use general knowledge.
- **PDF error:** Use a readable text-based PDF. Scanned image PDFs require OCR first.
- **Slow first run:** Sentence Transformers downloads and caches the embedding model once.
- **Old data appears:** Click **Clear Vector Database** and restart the app; the bundled Sanjivani project data will index automatically.
