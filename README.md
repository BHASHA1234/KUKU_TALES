# KUKUTales AI Storyteller (Streamlit App)

This project is an AI-powered multi-chapter story generation pipeline implemented in Python. It uses Google's Gemini models for core text generation and embeddings, ChromaDB for vector-based memory, NetworkX for knowledge graph representation, gTTS for text-to-speech audio generation, and OpenAI's DALL-E 3 for cover image generation. The user interface is built with Streamlit, allowing for an interactive story creation process.

## Overview

The application takes a simple user concept and guides it through several stages:

1.  **Concept Expansion:** The initial idea is expanded by an LLM into a structured format (logline, characters, setting, conflict).
2.  **Outline Generation:** A multi-chapter outline proposal is generated based on the expanded concept.
3.  **Outline Review & Revision:** The user can review the proposed outline and optionally provide feedback for regeneration.
4.  **State Initialization:** Based on the confirmed outline and concept, the application initializes character states, lore, a knowledge graph (KG), and a vector database (ChromaDB) for Retrieval-Augmented Generation (RAG).
5.  **Chapter Generation:** The pipeline iteratively generates chapters, using the outline, current character/lore state, KG facts, and relevant context retrieved from the vector database (RAG) to inform the LLM.
6.  **State Updates:** After each chapter, the system analyzes the generated content to update character states, relationships, and knowledge in the KG and adds a summary to the vector database.
7.  **Final Output:** Presents the complete story chapter by chapter, generates full-story audio using gTTS, and optionally generates a book cover using DALL-E 3.

## Key Features

*   **LLM-Powered Generation:** Leverages Google Gemini models for concept expansion, outlining, narrative writing, summarization, and state analysis.
*   **Interactive Outlining:** Allows users to review and provide feedback on the generated story outline before proceeding.
*   **Persistent Memory:** Uses ChromaDB as a vector store to maintain context and retrieve relevant past events (RAG).
*   **Knowledge Graph:** Employs NetworkX to model character relationships, states, and possessions, providing structured context.
*   **State Tracking:** Dynamically updates character states and lore based on generated chapter content.
*   **Text-to-Speech:** Generates an MP3 audio file of the full story using gTTS.
*   **Cover Art Generation:** Creates a unique book cover image based on the story concept using OpenAI's DALL-E 3 API.
*   **Streamlit Frontend:** Provides an easy-to-use web interface for interacting with the pipeline.

## Technologies Used

*   **Backend:** Python 3.x
*   **Frontend:** Streamlit
*   **LLMs & Embeddings:** Google Generative AI (Gemini Pro/Flash)
*   **Image Generation:** OpenAI API (DALL-E 3)
*   **Vector Database:** ChromaDB
*   **Knowledge Graph:** NetworkX
*   **Text-to-Speech:** gTTS (Google Text-to-Speech)
*   **API Key Management:** python-dotenv
*   **HTTP Requests:** requests (for image download)

## Setup and Installation

1.  **Prerequisites:**
    *   Python 3.8+
    *   pip (Python package installer)
    *   Git (for cloning the repository)

2.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    # Example: git clone https://github.com/YourNewUsername/ai-storyteller-project.git
    # Example: cd ai-storyteller-project
    ```

3.  **Create and Activate a Virtual Environment (Recommended):**
    *   **macOS/Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    *   **Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure you have a `requirements.txt` file listing all necessary packages: streamlit, google-generativeai, chromadb, networkx, python-dotenv, gTTS, openai, requests)*

5.  **Configure API Keys:**
    *   Create a file named `.env` in the root project directory.
    *   Add your API keys to the `.env` file in the following format:
        ```plaintext
        GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY_HERE
        OPENAI_API_KEY=YOUR_OPENAI_API_KEY_HERE
        ```
    *   Replace `YOUR_GOOGLE_API_KEY_HERE` with your actual Google AI Studio API key.
    *   Replace `YOUR_OPENAI_API_KEY_HERE` with your actual OpenAI API key.
    *   **IMPORTANT:** Do NOT commit the `.env` file to Git. The included `.gitignore` file should prevent this automatically.

## Usage

1.  **Activate Virtual Environment:** Make sure your virtual environment is active (you should see `(venv)` in your terminal prompt).
    ```bash
    # Example for macOS/Linux
    source venv/bin/activate
    ```

2.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```

3.  **Interact with the App:**
    *   Streamlit will open the application in your web browser.
    *   Enter your initial story concept in the text area and click "Generate Outline Proposal".
    *   Review the generated outline. Provide feedback and click "Revise Outline" or click "Accept & Write" to proceed.
    *   The app will then initialize the story state and generate chapters one by one, displaying logs.
    *   Once finished, you can view the generated chapters, generate/listen to the audio, and generate the book cover.
    *   Click "Start New Story" to reset and begin again.

## File Structure
.
├── app.py # Streamlit frontend application
├── pipeline_logic.py # Backend logic for story generation pipeline steps
├── requirements.txt # Project dependencies
├── .gitignore # Specifies intentionally untracked files for Git
├── .env # API Keys (!!! IMPORTANT: Create locally, DO NOT COMMIT !!!)
├── images/ # Optional: Directory for source images like logos
│ └── kukutales_logo.png # Example logo
├── chroma_story_db/ # Directory for ChromaDB persistent storage (created automatically)
├── generated_story_gtts.mp3 # Example generated audio output (ignored by .gitignore)
├── generated_book_cover.png # Example generated image output (ignored by .gitignore)
└── README.md # This file
## Future Improvements

*   More sophisticated Knowledge Graph reasoning and updates.
*   Ability to save and load story generation progress.
*   Support for different LLMs or TTS providers.
*   More fine-grained control over generation parameters (temperature, style).
*   Enhanced UI/UX with visualizations of the KG or character timelines.
*   More robust error handling and user feedback.
