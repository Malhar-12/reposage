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
    ".yml", ".toml", ".sh", ".sql", ".graphql", ".html", ".css",
}

SKIP_PATTERNS = [
    "package-lock.json", "yarn.lock", "poetry.lock", ".min.js",
    ".min.css", "dist/", "build/", "node_modules/", "__pycache__/",
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
        self.indexes = {}    # repo_id -> {bm25, docs, paths}
        self.collections = {}  # kept for compatibility with main.py

    def _parse_url(self, url: str) -> tuple:
        m = re.search(r"github\.com/([^/\s]+)/([^/\s]+?)(?:\.git|/?$)", url.strip())
        if not m:
            raise ValueError("Invalid GitHub URL. Expected: https://github.com/owner/repo")
        return m.group(1), m.group(2).rstrip("/")

    async def _fetch_files(self, owner: str, repo: str) -> list:
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "RepoSage/1.0"}
        if token := os.getenv("GITHUB_TOKEN"):
            headers["Authorization"] = f"Bearer {token}"

        async with httpx.AsyncClient(timeout=30) as client:
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
            paths = []
            for item in tree:
                if item["type"] != "blob":
                    continue
                path = item["path"]
                ext = os.path.splitext(path)[1].lower()
                size = item.get("size", 0)
                if ext not in ALLOWED_EXT:
                    continue
                if size > 50_000 or size == 0:
                    continue
                if any(skip in path for skip in SKIP_PATTERNS):
                    continue
                paths.append(path)

            # Prioritise root and shallow files, cap at 40
            paths.sort(key=lambda p: (p.count("/"), p))
            paths = paths[:40]

            results = []
            for i in range(0, len(paths), 10):
                batch = paths[i:i + 10]
                tasks = [
                    client.get(
                        f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{p}",
                        timeout=10,
                    )
                    for p in batch
                ]
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                for path, resp in zip(batch, responses):
                    if isinstance(resp, Exception):
                        continue
                    if resp.status_code == 200:
                        results.append({"path": path, "content": resp.text[:8000]})

            return results

    def _chunk(self, path: str, content: str) -> list:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".py":
            parts = re.split(r"\n(?=(?:def |class |async def ))", content)
        elif ext in {".js", ".ts", ".jsx", ".tsx"}:
            parts = re.split(r"\n(?=(?:function |class |export |const \w+ = ))", content)
        elif ext in {".md", ".txt"}:
            parts = re.split(r"\n(?=#{1,3} |\n\n)", content)
        else:
            parts = [content[i:i + 900] for i in range(0, len(content), 700)]

        chunks = []
        for part in parts:
            part = part.strip()
            if len(part) < 40:
                continue
            if len(part) > 1200:
                for j in range(0, len(part), 900):
                    sub = part[j:j + 1200].strip()
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
            raise ValueError("No indexable files found. Check the URL or try adding a GITHUB_TOKEN.")

        docs, paths = [], []
        for file in files:
            for chunk in self._chunk(file["path"], file["content"]):
                docs.append(f"# File: {file['path']}\n\n{chunk}")
                paths.append(file["path"])

        tokenized = [self._tokenize(doc) for doc in docs]
        bm25 = BM25Okapi(tokenized)

        self.indexes[repo_id] = {"bm25": bm25, "docs": docs, "paths": paths}
        self.collections[repo_id] = repo_id

        return {"repo_id": repo_id, "file_count": len(files), "chunk_count": len(docs)}

    async def chat_stream(self, repo_id: str, question: str, history: list) -> AsyncGenerator:
        index = self.indexes.get(repo_id)
        if not index:
            yield {"type": "error", "content": "Repository not indexed."}
            return

        tokens = self._tokenize(question)
        scores = index["bm25"].get_scores(tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]

        context_parts, sources, seen = [], [], set()
        for i in top_idx:
            context_parts.append(index["docs"][i])
            if index["paths"][i] not in seen:
                sources.append(index["paths"][i])
                seen.add(index["paths"][i])

        system = SYSTEM_PROMPT.format(context="\n\n---\n\n".join(context_parts))
        messages = list(history[-6:])
        messages.append({"role": "user", "content": question})

        with self.ai.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield {"type": "text", "content": text}

        yield {"type": "sources", "content": sources}