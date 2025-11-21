🎹 ManualMaster

<div align="center">

RAG-Chatbot for Musicians: Instrument Manuals & Workflow Setup

An AI-powered assistant for gear functionality, routing, and studio workflow optimization

Overview • Architecture • Installation • Usage • Structure

</div>

🎯 Overview

Music Gear Chat is a specialized RAG (Retrieval-Augmented Generation) chatbot designed to solve the "manual fatigue" problem for musicians and producers. Instead of CTRL+F searching through dense PDF manuals, users can query their gear in natural language.

It specializes in Elektron workflows but is adaptable to any equipment with a PDF manual.

Why Music Gear Chat?

While standard LLMs have general knowledge, this project grounds answers in specific technical documentation:

Component

Purpose

Benefit

🧩 RAG Pipeline

Retrieves exact manual pages

Eliminates hallucinations about button combos

⚡ Streamlit UI

Interactive Chat Interface

accessible, clean, and responsive user experience

🔗 Elektron Auto-Fetch

Automated Manual Downloader

Instant setup for Octatrack, Digitakt, etc.

🎚️ Workflow Logic

Context-aware routing advice

Understands MIDI/Audio signal flow between units

🏗️ Architecture

Information Retrieval Flow

┌─────────────────────────────────────────────────────────────────┐
│                       Interaction Loop                          │
│  ┌──────┐    ┌───────────┐    ┌───────────┐    ┌──────────┐     │
│  │ User │───▶│ Streamlit │───▶│ LangChain │───▶│  OpenAI  │     │
│  │      │    │    UI     │    │   (RAG)   │    │   GPT-4  │     │
│  └──────┘    └───────────┘    └───────────┘    └──────────┘     │
│                                     │                 ▲         │
│                                     ▼                 │         │
│                              ┌─────────────┐          │         │
│                              │ Vector Store│──────────┘         │
│                              │ (Chroma/FAISS)                   │
│                              └─────────────┘                    │
└─────────────────────────────────────────────────────────────────┘


🚀 Installation

Prerequisites

# Python 3.8+
pip install streamlit langchain openai pypdf chromadb


1. Clone the Repository

git clone [https://github.com/Jada42/music-gear-chat.git](https://github.com/Jada42/music-gear-chat.git)
cd music-gear-chat


2. Environment Setup

It is recommended to use a virtual environment.

python -m venv venv
# Windows: .\venv\Scripts\activate
# Mac/Linux: source venv/bin/activate


3. Install Dependencies

pip install -r requirements.txt


⚙️ Configuration

Create a .env file in the root directory to store your API credentials.

OPENAI_API_KEY=sk-your_api_key_here


🎛️ Usage

Phase 1: Data Ingestion

Before chatting, populate your knowledge base. You can manually add PDFs to the manuals/ folder, or use the included utility for Elektron gear:

python download_elektron_manuals.py


Phase 2: Launch Interface

Start the Streamlit application:

streamlit run app.py


The application will open in your default browser at http://localhost:8501.

📂 Project Structure

music-gear-chat/
├── app.py                         # Main Streamlit Chat Interface
├── download_elektron_manuals.py   # Utility: Auto-fetch Elektron PDFs
├── requirements.txt               # Project Dependencies
├── manuals/                       # PDF Storage Directory
└── .devcontainer/                 # VS Code Dev Container Config


🔮 Future Directions

[ ] Support for non-Elektron manufacturers (Roland, Korg, etc.)

[ ] Local LLM support (Llama 3 / Mistral) for offline use

[ ] Image recognition for front-panel settings

[ ] Audio-in analysis for patch debugging

🤝 Contributing

Contributions are welcome! Please follow these steps:

Fork the repository

Create a feature branch

Commit your changes

Open a Pull Request

📄 License

Distributed under the MIT License. See LICENSE for more information.

<div align="center">

Built by Jada42

⬆ Back to Top

</div>
