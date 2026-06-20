import os
import re
import httpx
import chromadb
import anthropic
from typing import AsyncGenerator

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Pre-load embedding model at startup so first request doesn't time out
from chromadb.utils import embedding_functions
_ef = embedding_functions.DefaultEmbeddingFunction()
print("Embedding model loaded successfully")

# File types we want to index
ALLOWED_EXT = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
    ".cpp", ".c", ".h", ".cs", ".rb", ".php", ".swift", ".kt",
    ".md", ".txt", ".yaml", ".yml", ".toml", ".sh", ".env.example",
    ".sql", ".graphql", ".proto", ".html", ".css",
}

# Files to skip even if extension matches
SKIP_PATTERNS = {
    "package-lock.json", "yarn.lock", "poetry.lock", "Pipfile.lock",
    ".min.js", ".min.css", "dist/", "build/", "node_modules/",
    "__pycache__/", ".git/",
}

SYSTEM_PROMPT = """You are an expert software engineer helping developers understand a GitHub codebase.
Use the retrieved code snippets below to answer the question accurately.
Always reference specific file paths (e.g., `src/auth/jwt.py`) when explaining code.
Format code examples with markdown code blocks and the correct language tag.
If the retrieved context doesn't contain enough to answer, say so clearly.

RETRIEVED CODE CONTEXT:
{context}"""


class RAGSystem:
    def __init__(self):
        self.chroma = chromadb.Client()  # in-memory; swap for chromadb.PersistentClient() in prod
        self.ai = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.collections: dict[str, str] = {}  # repo_id -> collection name

    # ── URL parsing ──────────────────────────────────────────────────────────

    def _parse_url(self, url: str) -> tuple[str, str]:
        m = re.search(r"github\.com/([^/\s]+)/([^/\s]+?)(?:\.git|/?$)", url.strip())
        if not m:
            raise ValueError("Invalid GitHub URL. Expected: https://github.com/owner/repo")
        return m.group(1), m.group(2).rstrip("/")

    # ── GitHub file fetching ─────────────────────────────────────────────────

    async def _fetch_files(self, owner: str, repo: str) -> list[dict]:
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "RepoSage/1.0"}
        if token := os.getenv("GITHUB_TOKEN"):
            headers["Authorization"] = f"Bearer {token}"

        async with httpx.AsyncClient(timeout=30) as client:
            # Try default branch detection first
            branch = "HEAD"
            for b in ("main", "master", "HEAD"):
                r = await client.get(
                    f"https://api.github.com/repos/{owner}/{repo}/git/trees/{b}?recursive=1",
                    headers=headers,
                )
                if r.status_code == 200:
                    branch = b
                    break

            if r.status_code != 200:
                raise ValueError(f"Cannot access repo {owner}/{repo}. Is it public?")

            tree = r.json().get("tree", [])

            # Filter files
            paths = []
            for item in tree:
                if item["type"] != "blob":
                    continue
                path = item["path"]
                ext = os.path.splitext(path)[1].lower()
                size = item.get("size", 0)

                if ext not in ALLOWED_EXT:
                    continue
                if size > 80_000 or size == 0:
                    continue
                if any(skip in path for skip in SKIP_PATTERNS):
                    continue
                paths.append(path)

            # Prioritise: root-level > src/ > fewer subdirectory levels; cap at 80
            paths.sort(key=lambda p: (p.count("/"), p))
            paths = paths[:80]

            # Fetch content in parallel (batches of 10)
            results = []
            for i in range(0, len(paths), 10):
                batch = paths[i : i + 10]
                tasks = [
                    client.get(
                        f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{p}",
                        headers={"Accept": "text/plain"},
                        timeout=10,
                    )
                    for p in batch
                ]
                import asyncio
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                for path, resp in zip(batch, responses):
                    if isinstance(resp, Exception):
                        continue
                    if resp.status_code == 200:
                        results.append({"path": path, "content": resp.text[:12_000]})

            return results

    # ── Smart chunking ───────────────────────────────────────────────────────

    def _chunk(self, path: str, content: str) -> list[str]:
        ext = os.path.splitext(path)[1].lower()
        chunks = []

        if ext in {".py"}:
            # Split on top-level def / class boundaries
            parts = re.split(r"\n(?=(?:def |class |async def ))", content)
        elif ext in {".js", ".ts", ".jsx", ".tsx"}:
            # Split on function / class / export keywords
            parts = re.split(r"\n(?=(?:function |class |export |const \w+ = (?:async )?(?:\([^)]*\)|[^=]+)=>))", content)
        elif ext in {".java", ".go", ".cs", ".kt", ".swift"}:
            # Split on method/function definitions
            parts = re.split(r"\n(?=(?:public |private |protected |func |fun ))", content)
        elif ext in {".md", ".txt"}:
            # Split on headings or double newlines
            parts = re.split(r"\n(?=#{1,3} |\n\n)", content)
        else:
            # Generic: fixed windows with overlap
            parts = []
            step = 900
            for i in range(0, len(content), step):
                parts.append(content[i : i + 1100])

        for part in parts:
            part = part.strip()
            if len(part) < 40:
                continue
            # Break oversized chunks further
            if len(part) > 1400:
                for j in range(0, len(part), 1100):
                    sub = part[j : j + 1400].strip()
                    if sub:
                        chunks.append(sub)
            else:
                chunks.append(part)

        return chunks

    # ── Index a repository ───────────────────────────────────────────────────

    async def index_repository(self, github_url: str) -> dict:
        owner, repo = self._parse_url(github_url)
        repo_id = f"{owner}/{repo}"

        files = await self._fetch_files(owner, repo)
        if not files:
            raise ValueError("No indexable files found. Check the URL or try adding a GITHUB_TOKEN.")

        # Safe ChromaDB collection name
        coll_name = re.sub(r"[^a-zA-Z0-9_-]", "_", repo_id)[:50]

        try:
            self.chroma.delete_collection(coll_name)
        except Exception:
            pass

        collection = self.chroma.create_collection(coll_name)

        docs, metas, ids = [], [], []
        for file in files:
            for chunk in self._chunk(file["path"], file["content"]):
                docs.append(f"# File: {file['path']}\n\n{chunk}")
                metas.append({"path": file["path"]})
                ids.append(f"{coll_name}_{len(ids)}")

        # Add in batches of 100
        for i in range(0, len(docs), 100):
            collection.add(
                documents=docs[i : i + 100],
                metadatas=metas[i : i + 100],
                ids=ids[i : i + 100],
            )

        self.collections[repo_id] = coll_name
        return {"repo_id": repo_id, "file_count": len(files), "chunk_count": len(docs)}

    # ── Chat with streaming ──────────────────────────────────────────────────

    async def chat_stream(
        self, repo_id: str, question: str, history: list
    ) -> AsyncGenerator[dict, None]:
        coll_name = self.collections.get(repo_id)
        if not coll_name:
            yield {"type": "error", "content": "Repository not indexed."}
            return

        collection = self.chroma.get_collection(coll_name)

        # Retrieve top-5 most relevant chunks
        results = collection.query(query_texts=[question], n_results=5)

        context_parts = []
        sources = []
        seen = set()
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            context_parts.append(doc)
            if meta["path"] not in seen:
                sources.append(meta["path"])
                seen.add(meta["path"])

        context = "\n\n---\n\n".join(context_parts)
        system = SYSTEM_PROMPT.format(context=context)

        # Build conversation history (last 3 turns = 6 messages)
        messages = list(history[-6:])
        messages.append({"role": "user", "content": question})

        # Stream from Claude
        with self.ai.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield {"type": "text", "content": text}

        # Send sources after the answer
        yield {"type": "sources", "content": sources}