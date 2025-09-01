Enhanced Multimodal PDF RAG System
==================================

Overview
--------

This system is an advanced Retrieval-Augmented Generation (RAG) implementation designed for comprehensive analysis of PDF documents containing both textual and visual content. It combines multiple AI technologies to provide accurate, context-aware responses based strictly on document content.

Setup Instructions
------------------

### 1\. Clone and Navigate

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`git clone   cd` 

### 2\. Create and Activate Virtual Environment

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python3 -m venv venv  source venv/bin/activate   `

### 3\. Install Dependencies

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   pip3 install -r requirements.txt   `

### 4\. Environment Configuration

*   Create a .env file in the project root.
    
*   OPENAI\_API\_KEY=your\_api\_key\_here
    
*   pdf\_path = "your\_pdf\_path\_here"
    

Running the Server
------------------

Start the Flask server on **localhost:5001**:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python app.py   `

API Endpoints
-------------

### Ask a Question

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   curl -X POST http://127.0.0.1:5001/ask \    -H "Content-Type: application/json" \    -d '{"question": "What are the key market trends?"}'   `

### System Health Check

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   curl http://127.0.0.1:5001/health   `

### Debug Retrieval

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   curl -X POST http://127.0.0.1:5001/debug \    -H "Content-Type: application/json" \    -d '{"question": "Compare software vs hardware growth rates"}'   `

Architecture Overview
---------------------

### Core Technologies

*   **Document Processing & Extraction**: PyMuPDF (fitz)
    
*   **Embeddings**:
    
    *   Text: sentence-transformers/msmarco-distilbert-dot-v5
        
    *   Images: openai/clip-vit-base-patch32
        
*   **Vector Storage**: FAISS
    
*   **Reranker**: BAAI/bge-reranker-base
    
*   **LLMs**: GPT-4o (primary), GPT-3.5-turbo (fallback)
    
*   **Framework**: Flask REST API
    

### Key Features

*   Document-only responses with **strict source attribution** (page-level).
    
*   **Multimodal understanding** (text + images).
    
*   **Efficient retrieval** with FAISS and reranking.
    
*   **Scalable** and optimized for professional document analysis.
    

Example Workflow
----------------

1.  **Initialization**
    
    *   Load API keys and models.
        
    *   Setup FAISS vector store.
        
2.  **Document Processing**
    
    *   Parse PDFs, extract text + images.
        
    *   Chunk text (chunk\_size=800, chunk\_overlap=200).
        
    *   Generate embeddings and store them.
        
3.  **Query Handling**
    
    *   Retrieve relevant text and image chunks.
        
    *   Apply reranking for precision.
        
    *   Construct ultra-strict prompt with sources.
        
4.  **Response Generation**
    
    *   LLM generates answer with citations.
        
    *   Fallback mechanisms ensure reliability.
        

Configuration Parameters
------------------------

*   **Chunk Size**: 800 characters
    
*   **Overlap**: 200 characters
    
*   **Retrieval**: Top 25 candidates → rerank to 5
    
*   **LLM Temperature**: 0.1
    
*   **Image Limit**: 2 per query
    

Conclusion
----------

The **Enhanced Multimodal PDF RAG System** is a state-of-the-art solution for PDF document analysis. By combining advanced embeddings, multimodal retrieval, reranking, and LLMs, it ensures **highly accurate, source-grounded answers** suitable for professional and enterprise use cases.
