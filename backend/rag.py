import os
import re
import asyncio
import httpx
import anthropic
from rank_bm25 import BM25Okapi
from typing import AsyncGenerator

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

ALLOWED_EXT = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
    ".cpp", ".c", ".h", ".cs", ".rb", ".md", ".txt", ".yaml",
    ".yml", ".toml", ".sh", ".sql", ".html", ".css",
}

SKIP_PATTERNS = [
    "package-lock.json", "yarn.lock", "poetry.lock", "Pipfile.lock",
    ".min.js", ".min.css", "dist/", "build/", "node_modules/",
    "__pycache__/", ".git/", "vendor/", "coverage/", ".next/",
    "migrations/", "static/", "assets/", "public/images/",
]

SYSTEM_PROMPT = """You are an expert software engineer helping developers understand a GitHub codebase.
Use the retrieved code snippets below to answer accurately.
Always reference specific file paths when explaining code.
Format code with markdown code blocks and the correct language tag.
If the context is not enough to answer, say so clearly.

RETRIEVED CODE CONTEXT:
{context}"""


class RAGSystem:
    def __init__(self):
        self.ai = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.indexes = {}      # repo_id -> {bm25, docs, paths}
        self.collections = {}  # kept for compatibility with main.py

    def _parse_url(self, url: str) -> tuple:
        m = re.search(r"github\.com/([^/\s]+)/([^/\s]+?)(?:\.git|/?$)", url.strip())
        if not m:
            raise ValueError("Invalid GitHub URL. Expected: https://github.com/owner/repo")
        return m.group(1), m.group(2).rstrip("/")

    async def _fetch_files(self, owner: str, repo: str) -> list:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "RepoSage/1.0",
        }
        if token := os.getenv("GITHUB_TOKEN"):
            headers["Authorization"] = f"Bearer {token}"

        # Use a longer timeout since we're running as background task
        async with httpx.AsyncClient(timeout=60) as client:
            # Detect default branch
            branch = "main"
            for b in ("main", "master", "HEAD"):
                try:
                    r = await client.get(
                        f"https://api.github.com/repos/{owner}/{repo}/git/trees/{b}?recursive=1",
                        headers=headers,
                    )
                    if r.status_code == 200:
                        branch = b
                        break
                except Exception:
                    continue
            else:
                raise ValueError(
                    f"Cannot access repo {owner}/{repo}. "
                    "Check if it's public or if your GITHUB_TOKEN is valid."
                )

            tree = r.json().get("tree", [])
            if not tree:
                raise ValueError(f"Repository {owner}/{repo} appears to be empty.")

            # Score and filter files
            scored = []
            for item in tree:
                if item["type"] != "blob":
                    continue
                path = item["path"]
                ext = os.path.splitext(path)[1].lower()
                size = item.get("size", 0)

                if ext not in ALLOWED_EXT:
                    continue
                if size > 40_000 or size == 0:
                    continue
                if any(skip in path for skip in SKIP_PATTERNS):
                    continue

                # Score: prefer shallow, important files
                depth = path.count("/")
                is_important = any(
                    name in path.lower()
                    for name in ["main", "app", "index", "core", "api", "server",
                                 "readme", "config", "router", "model", "auth"]
                )
                score = depth * 10 - (5 if is_important else 0)
                scored.append((score, path))

            # Sort by score, take top 15 files only (fast + reliable)
            scored.sort()
            paths = [p for _, p in scored[:15]]

            if not paths:
                raise ValueError("No indexable source files found in this repository.")

            # Fetch all files in parallel
            tasks = [
                client.get(
                    f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{p}",
                    timeout=15,
                )
                for p in paths
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            results = []
            for path, resp in zip(paths, responses):
                if isinstance(resp, Exception):
                    continue
                if resp.status_code == 200 and resp.text.strip():
                    results.append({
                        "path": path,
                        "content": resp.text[:6000],  # cap per file
                    })

            return results

    def _chunk(self, path: str, content: str) -> list:
        ext = os.path.splitext(path)[1].lower()

        if ext == ".py":
            parts = re.split(r"\n(?=(?:def |class |async def ))", content)
        elif ext in {".js", ".ts", ".jsx", ".tsx"}:
            parts = re.split(r"\n(?=(?:function |class |export |const \w+ = ))", content)
        elif ext in {".md", ".txt"}:
            parts = re.split(r"\n(?=#{1,3} )", content)
        else:
            # Fixed-size windows with overlap
            parts = []
            step = 600
            for i in range(0, len(content), step):
                parts.append(content[i:i + 900])

        chunks = []
        for part in parts:
            part = part.strip()
            if len(part) < 20:
                continue
            if len(part) > 1000:
                for j in range(0, len(part), 800):
                    sub = part[j:j + 1000].strip()
                    if sub:
                        chunks.append(sub)
            else:
                chunks.append(part)
        return chunks

    def _tokenize(self, text: str) -> list:
        return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower())

    async def index_repository(self, github_url: str) -> dict:
        owner, repo = self._parse_url(github_url)
        repo_id = f"{owner}/{repo}"

        files = await self._fetch_files(owner, repo)
        if not files:
            raise ValueError(
                "Could not fetch any files from this repository. "
                "Make sure it's a public repo and try again."
            )

        docs, paths = [], []
        for file in files:
            for chunk in self._chunk(file["path"], file["content"]):
                docs.append(f"# File: {file['path']}\n\n{chunk}")
                paths.append(file["path"])

        if not docs:
            raise ValueError("Files were fetched but no content could be extracted.")

        tokenized = [self._tokenize(doc) for doc in docs]
        bm25 = BM25Okapi(tokenized)

        self.indexes[repo_id] = {"bm25": bm25, "docs": docs, "paths": paths}
        self.collections[repo_id] = repo_id

        return {
            "repo_id": repo_id,
            "file_count": len(files),
            "chunk_count": len(docs),
        }

    async def chat_stream(
        self, repo_id: str, question: str, history: list
    ) -> AsyncGenerator:
        index = self.indexes.get(repo_id)
        if not index:
            yield {"type": "error", "content": "Repository not indexed. Please index it first."}
            return

        # BM25 retrieval
        tokens = self._tokenize(question)
        scores = index["bm25"].get_scores(tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]

        context_parts, sources, seen = [], [], set()
        for i in top_idx:
            if scores[i] > 0:  # only include relevant chunks
                context_parts.append(index["docs"][i])
                if index["paths"][i] not in seen:
                    sources.append(index["paths"][i])
                    seen.add(index["paths"][i])

        # Fallback: if nothing scored, just use top 3
        if not context_parts:
            for i in top_idx[:3]:
                context_parts.append(index["docs"][i])
                if index["paths"][i] not in seen:
                    sources.append(index["paths"][i])
                    seen.add(index["paths"][i])

        system = SYSTEM_PROMPT.format(context="\n\n---\n\n".join(context_parts))
        messages = list(history[-6:])
        messages.append({"role": "user", "content": question})

        try:
            with self.ai.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=system,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    yield {"type": "text", "content": text}
        except anthropic.AuthenticationError:
            yield {"type": "error", "content": "Invalid API key. Please check your ANTHROPIC_API_KEY."}
            return
        except anthropic.RateLimitError:
            yield {"type": "error", "content": "Rate limit hit. Please wait a moment and try again."}
            return
        except Exception as e:
            yield {"type": "error", "content": f"AI error: {str(e)}"}
            return

        yield {"type": "sources", "content": sources}