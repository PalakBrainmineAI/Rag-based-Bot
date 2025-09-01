# Import all libraries
import fitz, os, base64, io, warnings
from langchain_core.documents import Document
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch, numpy as np
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import openai
from openai import OpenAI
import time

# New imports for Cross Encoder Reranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from flask import Flask, request, jsonify, send_from_directory

import os
from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings('ignore')

print("✅ All packages installed and imported!")

# Setup OpenAI API key with validation (non-interactive)
def setup_openai_key():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY is not set in environment. Set it before starting the server.")
        return False

    os.environ["OPENAI_API_KEY"] = api_key

    try:
        client = OpenAI(api_key=api_key)
        client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        print("✅ API key validated successfully!")
        return True
    except Exception as e:
        print(f"⚠️ API key validation warning: {e}")
        # Continue; LangChain's client may still handle auth if key is valid
        return True

# Setup API key (non-blocking)
_ = setup_openai_key()

# Initialize CLIP model for images
print("🔄 Loading CLIP model...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()
print("✅ CLIP model loaded!")

# Initialize text embeddings model for better semantic search
print("🔄 Loading text embedding model...")
text_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/msmarco-distilbert-dot-v5"
)
print("✅ Text embedding model loaded!")

# Initialize Cross Encoder Reranker
print("🔄 Loading Cross Encoder Reranker...")
cross_encoder_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
reranker = CrossEncoderReranker(model=cross_encoder_model, top_n=5)
print("✅ Cross Encoder Reranker loaded!")

# Define embedding functions
def embed_image(image_data):
    if isinstance(image_data, str):
        image = Image.open(image_data).convert("RGB")
    else:
        image = image_data
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()

def embed_text_clip(text):
    inputs = clip_processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=77)
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()

# Skip file upload if already processed
if 'all_docs' not in globals():
    # Upload PDF file
    print("📁 Please upload your PDF file:")
    pdf_path = r"/Users/palaktiwari/Downloads/RAG-PDF-main/Report_Global Outdoor Air Quality Monitoring System Market.pdf"
    print(f"✅ Uploaded: {pdf_path}")

    # Process PDF
    print("🔄 Processing PDF...")
    doc = fitz.open(pdf_path)

    text_docs, image_docs = [], []
    image_embeddings, image_data_store = [], {}

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)

    for i, page in enumerate(doc):
        if i % 50 == 0:
            print(f"Processing page {i+1}...")

        # Process text
        text = page.get_text()
        if text.strip():
            temp_doc = Document(
                page_content=text,
                metadata={"page": i, "type": "text", "source": pdf_path}
            )
            text_chunks = splitter.split_documents([temp_doc])
            text_docs.extend(text_chunks)

        # Process images
        for img_index, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                image_id = f"page_{i}img{img_index}"

                # Store as base64
                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                image_data_store[image_id] = img_base64

                # Create embedding and document
                embedding = embed_image(pil_image)
                image_embeddings.append(embedding)

                # Create a more descriptive placeholder for image content
                image_description = f"Image from page {i+1} - visual content that may contain charts, diagrams, illustrations, or other graphical information relevant to the document."
                image_doc = Document(
                    page_content=image_description,
                    metadata={"page": i, "type": "image", "image_id": image_id, "source": pdf_path}
                )
                image_docs.append(image_doc)
            except Exception as e:
                continue

    doc.close()
    print(f"✅ Processing complete! Text docs: {len(text_docs)}, Image docs: {len(image_docs)}")

    # Create separate vector stores for text and images
    print("🔄 Creating vector stores...")

    # Text vector store with better embeddings
    text_vector_store = FAISS.from_documents(text_docs, text_embeddings)

    # Image vector store (using CLIP embeddings)
    if image_docs:
        # Create a dummy embedding function for FAISS compatibility
        class DummyEmbedding:
            def embed_documents(self, texts):
                return [np.zeros(512) for _ in texts]
            def embed_query(self, text):
                return embed_text_clip(text)

        image_vector_store = FAISS.from_embeddings(
            text_embeddings=[(doc.page_content, emb) for doc, emb in zip(image_docs, image_embeddings)],
            embedding=DummyEmbedding(),
            metadatas=[doc.metadata for doc in image_docs]
        )
    else:
        image_vector_store = None

    print("✅ Vector stores created!")
else:
    print("📋 Using previously processed documents...")

# Initialize LLM
def create_llm():
    try:
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,  # Slightly higher for more natural responses
            timeout=60,
            max_retries=3
        )
        print("✅ GPT-4o model initialized!")
        return llm
    except Exception as e:
        print(f"❌ Failed to initialize GPT-4o: {e}")
        try:
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.1,
                timeout=60,
                max_retries=3
            )
            print("✅ GPT-3.5-turbo model initialized (fallback)!")
            return llm
        except Exception as e2:
            print(f"❌ Failed to initialize any model: {e2}")
            return None

llm = create_llm()

# Enhanced retrieval with reranking
def enhanced_retrieve(query, k=15):
    # Retrieve from text documents
    text_retriever = text_vector_store.as_retriever(search_kwargs={"k": k})

    # Create compression retriever with reranker
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=text_retriever
    )

    # Get reranked text documents
    reranked_text_docs = compression_retriever.invoke(query)

    # Retrieve relevant images if available
    image_results = []
    if image_vector_store:
        try:
            query_embedding = embed_text_clip(query)
            image_results = image_vector_store.similarity_search_by_vector(
                embedding=query_embedding, k=5
            )
        except:
            pass

    return reranked_text_docs, image_results

def create_context_aware_prompt(query, text_docs, image_docs):
    """Create a prompt that ensures answers come only from the document"""

    context_parts = []

    # Add text context
    if text_docs:
        context_parts.append("TEXT CONTENT FROM DOCUMENT:")
        for i, doc in enumerate(text_docs, 1):
            page_info = f"[Page {doc.metadata.get('page', 'unknown')}]"
            context_parts.append(f"{page_info}: {doc.page_content}")

    # Add image context
    if image_docs:
        context_parts.append("\nIMAGE CONTENT FROM DOCUMENT:")
        for doc in image_docs:
            page_info = f"[Page {doc.metadata.get('page', 'unknown')}]"
            context_parts.append(f"{page_info}: {doc.page_content}")

    context = "\n\n".join(context_parts)

    # Create a strict prompt that ensures document-only answers
    prompt = f"""You are an expert document analyst. Your task is to answer questions based STRICTLY on the provided document content.

IMPORTANT RULES:
1. ONLY use information that is explicitly present in the provided document content
2. If the information is not in the document, clearly state "This information is not available in the provided document"
3. Answer in a natural, human-like way as if you've carefully read and understood the document
4. Cite specific page numbers when available
5. If there are relevant images mentioned, acknowledge them in your response

DOCUMENT CONTENT:
{context}

QUESTION: {query}

ANSWER (based only on the document content):"""

    return prompt

def create_multimodal_message_enhanced(query, text_docs, image_docs, include_images=True):
    """Enhanced message creation for multimodal responses"""

    # Start with the context-aware prompt
    prompt_text = create_context_aware_prompt(query, text_docs, image_docs)

    content = [{"type": "text", "text": prompt_text}]

    # Add images if available and using vision model
    if include_images and llm and llm.model_name == "gpt-4o" and image_docs:
        for doc in image_docs[:2]:  # Limit to 2 images
            image_id = doc.metadata.get("image_id")
            if image_id and image_id in image_data_store:
                content.append({
                    "type": "text",
                    "text": f"\n[Visual content from page {doc.metadata.get('page', 'unknown')}]:"
                })
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_data_store[image_id]}"}
                })

    return HumanMessage(content=content)

def enhanced_pdf_rag_pipeline(query, max_retries=3):
    """Enhanced RAG pipeline with reranking and document-strict responses"""

    if not llm:
        return "❌ LLM not initialized. Please check your API key."

    try:
        # Retrieve and rerank documents
        text_docs, image_docs = enhanced_retrieve(query, k=15)

        print(f"\n📋 Retrieved and reranked documents:")
        print(f"  - 📄 {len(text_docs)} text chunks (reranked)")
        print(f"  - 🖼 {len(image_docs)} images")

        if not text_docs and not image_docs:
            return "❌ No relevant content found in the document for your query."

        # Create enhanced message
        for attempt in range(max_retries):
            try:
                print(f"🔄 Generating response (attempt {attempt + 1})...")

                include_images = llm.model_name == "gpt-4o"
                message = create_multimodal_message_enhanced(query, text_docs, image_docs, include_images)

                response = llm.invoke([message])

                # Add source information
                pages_referenced = set()
                for doc in text_docs + image_docs:
                    if 'page' in doc.metadata:
                        pages_referenced.add(doc.metadata['page'])

                source_info = f"\n\n📚 Sources: Pages {', '.join(map(str, sorted(pages_referenced)))}" if pages_referenced else ""

                return response.content + source_info

            except Exception as e:
                print(f"❌ Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    print("⏳ Retrying in 2 seconds...")
                    time.sleep(2)
                else:
                    return f"❌ All attempts failed. Last error: {e}"

    except Exception as e:
        return f"❌ Error in retrieval process: {e}"

# Helper function for pretty printing documents (for debugging)
def pretty_print_docs(docs, title="Documents"):
    print(f"\n{title}:")
    print("=" * 100)
    for i, doc in enumerate(docs):
        page_info = f"Page {doc.metadata.get('page', 'unknown')}"
        doc_type = doc.metadata.get('type', 'unknown')
        print(f"\nDocument {i+1} ({doc_type}, {page_info}):")
        print("-" * 50)
        print(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
    print("=" * 100)

print("🎯 Enhanced system ready with Cross Encoder Reranking!")

# --- Minimal Web Server to interact with the RAG system ---
app = Flask(__name__, static_folder='.', static_url_path='')

@app.get('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.post('/ask')
def ask_question():
    data = request.get_json(silent=True) or {}
    question = (data.get('question') or '').strip()
    if not question:
        return jsonify({"error": "Question is required"}), 400

    answer = enhanced_pdf_rag_pipeline(question)
    return jsonify({"answer": answer})

@app.get('/health')
def health():
    status = {
        "status": "ok",
        "llm_initialized": bool(llm),
        "has_text_index": True,
        "has_image_index": bool('image_vector_store' in globals() and image_vector_store is not None)
    }
    return jsonify(status)

if __name__ == '__main__':
    # Start the web server without the reloader to avoid double-initialization
    app.run(host='127.0.0.1', port=2001, debug=False, use_reloader=False)