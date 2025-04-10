# 🧠 RAG AI Chat | Compare Embedding + LLMs

An interactive **RAG (Retrieval-Augmented Generation)** chatbot built with **Streamlit**, allowing users to upload their own PDF/TXT stories (fiction/non-fiction), and ask questions about the content. It supports flexible **embedding** options, **LLM model comparison**, **document reranking evaluation**, and even **Internet search** fallback using Tavily.

> Ideal for experimenting with different LLM + embedding combinations and evaluating their retrieval quality in context-aware Q&A.

---

## ✨ Features

- 📁 Upload and process **PDF/TXT** story files
- 🧠 Supports multiple **LLMs**: `OpenAI`, `Ollama (LLaMA3)`, `Gemini`, `Mistral`
- 🔍 Choose between **embedding backends**: `OpenAI`, `Ollama`, `HuggingFace`
- 🧩 Custom **chunk size** and **overlap** for document splitting
- 🔄 Compare LLM **outputs side-by-side** (same input, different engines)
- 🧠 **MMR Retriever** and **Cohere Reranker** supported
- 📊 Evaluation metrics: `Precision@k`, `Recall@k`, and `MAP`
- 🌐 Fallback to **Internet Search** (via Tavily) if needed
- 💬 Conversational chat interface using `st.chat_message` (Streamlit)

---

## 🛠️ Installation

### 1. Clone the repo

```bash
git clone https://github.com/yourname/myragchatbot.git
cd myragchatbot
```

### 2. Install dependencies

```bash
# Using Poetry (recommended)
poetry install

# Or using pip
pip install -r requirements.txt
```

### 3. Create a `.env` file

```env
OPEN_API_KEY=your-openai-api-key
TAVILY_API_KEY=your-tavily-api-key
GOOGLE_API_KEY=your-gemini-api-key
```

> Replace `your-xxx-api-key` with actual values.

### 4. (Optional) Set up local LLM

Install and run Ollama if you want to use local models like `llama3`.

```bash
brew install ollama
ollama run llama3
```

---

## 🚀 Running the App

```bash
streamlit run src/myragchatbot/app/main.py
```

Then open `http://localhost:8501` in your browser.

---

## 🧱 Project Structure

```
src/
├── myragchatbot/
│   ├── app/                  # Main Streamlit app
│   ├── core/                 # Query engine logic
│   ├── loaders/              # Loaders for PDF/TXT + DocumentProcessor
│   ├── llm_backends/         # Wrappers for OpenAI, Ollama, Gemini, Mistral
│   ├── embeddings/           # Embedder factory
│   ├── rerankers/            # Cohere-based reranker
│   ├── evaluation/           # Evaluation metrics (precision/recall/MAP)
vectorstore/                  # Chroma vector DBs (one per embedding)
data/                         # Uploaded files directory
```

---

## 🔍 Evaluation Metrics

After each query, if reranking is enabled:

- **Precision@k**: Proportion of top-k documents that are relevant.
- **Recall@k**: Proportion of all relevant documents that were retrieved.
- **MAP (Mean Average Precision)**: A summary score of ranked relevance.

All based on a relevance score threshold (default: `0.4`).

---

## 📸 Example Use Case

1. Upload a story in PDF or TXT
2. Click “Generate Vectorstore”
3. Ask questions like:
   - *"Who is the main character?"*
   - *"What happened in the ending?"*
   - *"What moral lesson can be drawn from this?"*
4. Select multiple LLMs to compare their answers side-by-side
5. Review chunking, reranked docs, and evaluation metrics

---

## ✅ To-Do / Improvements

- [ ] Add support for image-based PDFs (OCR fallback)
- [ ] UI improvement: table-based LLM comparison
- [ ] Highlight relevant document chunks in output
- [ ] Chat export (Markdown/PDF)
- [ ] Add benchmark mode for batch comparison
- [ ] Multi-language question answering
- [ ] Save/load sessions

---

## 💼 Credits & Tech Stack

Built with:

- **LangChain**
- **ChromaDB**
- **Streamlit**
- **Ollama**
- **Cohere**
- **Tavily**
- **Google Generative AI**
- **OpenAI GPT**

---

## 📜 License

MIT License  
© 2025 — Built with ❤️

---
