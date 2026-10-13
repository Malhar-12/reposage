# 🚀 RepoSage — Chat with Any GitHub Repository Using AI

**Stop digging through code. Just ask.**

RepoSage is an AI-powered developer tool that helps you understand any public GitHub repository using plain English. Paste a repository URL, ask questions about its codebase, and get context-aware answers with relevant file references.

🔗 **Live Demo:** [reposage-jjnm.onrender.com](https://reposage-jjnm.onrender.com)  
💻 **GitHub Repository:** [Malhar-12/reposage](https://github.com/Malhar-12/reposage)

---

## ✨ Features

- 🔗 **Any Public GitHub Repository** — Enter a public GitHub URL to explore its codebase.
- 💬 **Natural Language Questions** — Ask questions without manually searching through files.
- 🔍 **BM25 Search** — Retrieve relevant code chunks based on your query.
- 🧠 **AI-Powered Answers** — Use Claude AI to generate answers grounded in retrieved code.
- 📂 **Code References** — Find relevant files and code locations to help verify answers.
- ⚡ **Real-Time Streaming** — Receive responses as they are generated.
- 🧩 **Intelligent Code Chunking** — Organize code around function and class boundaries.
- 🌐 **No Repository Setup Required** — Start exploring without cloning the repository locally.

## 🎯 Why RepoSage?

Understanding an unfamiliar codebase can take hours or even days. RepoSage helps developers find relevant code and understand how different parts of a project work together.

Instead of manually searching through hundreds of files, you can ask questions like:

- How does request context work in Flask?
- Where is authentication implemented?
- How does the application handle API requests?
- Which file defines a particular function or class?
- How does data flow through this project?
- Where should I look to understand a specific feature?

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| FastAPI | Backend API and application server |
| BM25 | Relevant code retrieval |
| Claude API | AI-generated answers |
| GitHub API | Fetching repository source files |
| Streaming | Delivering responses in real time |

## ⚙️ How It Works

1. **Submit a repository URL** — Paste the URL of a public GitHub repository.
2. **Fetch source files** — Retrieve accessible source files using the GitHub API.
3. **Chunk the code** — Split source code into manageable sections, using function and class boundaries where supported.
4. **Search relevant code** — Use BM25 to identify the chunks most relevant to your question.
5. **Generate an answer** — Provide the retrieved context to Claude AI.
6. **Stream the response** — Return an explanation with relevant file references.

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or later
- A GitHub account or accessible public repositories
- An Anthropic API key

### 1. Clone the repository

```bash
git clone https://github.com/Malhar-12/reposage.git
cd reposage
```

### 2. Create a virtual environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root and add your API key:

```env
ANTHROPIC_API_KEY=your_anthropic_api_key
```

Keep your API key private. Never commit your `.env` file to GitHub.

### 5. Run the application

If your FastAPI entry point is `main.py`, run:

```bash
uvicorn main:app --reload
```

Open the local URL shown by your application, commonly `http://127.0.0.1:8000`.

> **Note:** Update the entry point, environment variable names, and startup command to match your actual project structure.

## 🧪 Example Usage

Try RepoSage with the Flask repository:

**Repository URL**

```text
https://github.com/pallets/flask
```

**Question**

```text
How does request context work in Flask?
```

RepoSage searches the codebase and returns an AI-generated explanation with relevant file references to help you explore the implementation.

## 👥 Who Is RepoSage For?

- **Developers joining a new team** — Get familiar with an unfamiliar codebase.
- **Open-source contributors** — Understand project architecture before contributing.
- **Students** — Learn how real-world software projects are structured.
- **Technical leads and reviewers** — Locate implementation details more quickly.
- **Engineers evaluating libraries** — Explore how a dependency works internally.

## 🔐 Security & Limitations

- RepoSage is designed to work with public GitHub repositories.
- Private repositories may not be accessible.
- Results depend on the source files retrieved and the relevance of the search results.
- AI-generated explanations can contain mistakes; verify important details against the original source code.
- API usage may incur costs depending on your Anthropic account and configuration.

## 🗺️ Future Improvements

Potential areas for future development include:

- Support for larger repositories and more programming languages.
- Improved code-aware retrieval and ranking.
- Repository architecture visualizations.
- Conversation history for each repository.
- More precise line-level code citations.
- Automated tests and expanded error handling.

## 🤝 Contributing

Contributions, suggestions, and bug reports are welcome!

1. Fork this repository.
2. Create a feature branch.
3. Make your changes.
4. Test your changes.
5. Submit a pull request describing your improvements.

## 👨‍💻 Author

**Malhar Kshirsagar**

- GitHub: [@Malhar-12](https://github.com/Malhar-12)
- Project: [RepoSage](https://github.com/Malhar-12/reposage)

Built independently with Python, FastAPI, BM25 search, and Claude AI.

## 📄 License

Add a `LICENSE` file to specify the terms under which others may use, modify, and distribute this project.

---

⭐ **If RepoSage helps you understand code faster, consider giving the repository a star!**

*Built for developers who want to spend less time searching and more time building.*
