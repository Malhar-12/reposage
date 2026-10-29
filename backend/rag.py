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
    ".yml", ".toml", ".sh", ".sql", ".html", ".css", ".vue", ".svelte",
}

SKIP_PATTERNS = [
    "package-lock.json", "yarn.lock", "poetry.lock", "Pipfile.lock",
    ".min.js", ".min.css", "dist/", "build/", "node_modules/",
    "__pycache__/", ".git/", "vendor/", "coverage/", ".next/",
    "migrations/", "static/", "assets/", "public/images/",
]

SYSTEM_PROMPT = """You are an expert software engineer helping developers understand a GitHub codebase.
Use the retrieved code snippets to answer accurately and concisely.
Always reference specific file paths (e.g. `src/auth/jwt.py`) when explaining code.
Format code examples with markdown code blocks and the correct language tag.
If context is insufficient, say so and suggest what to search for instead.

RETRIEVED CODE CONTEXT:
{context}"""

SUMMARY_PROMPT = """Analyze this GitHub repository's source files and provide a structured overview.

FILES CONTENT:
{context}

Provide:
1. **What this project does** (1-2 sentences)
2. **Main technologies** (list the stack)
3. **Key components** (3-5 main files/modules and what they do)
4. **Architecture pattern** (MVC, microservices, monolith, etc.)
5. **Entry point** (where execution starts)

Be concise and technical."""


class RAGSystem:
    def __init__(self):
        self.ai = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.indexes = {}
        self.collections = {}
        self.summaries = {}   # repo_id -> summary text
        self.file_lists = {}  # repo_id -> list of {path, size, ext}

    def _parse_url(self, url: str) -> tuple:
        url = url.strip()
        # Always extract just owner/repo from any GitHub URL
        m = re.search(r"github\.com/([^/\s]+)/([^/\s#?]+?)(?:\.git|/|$)", url)
        if not m:
            raise ValueError("Invalid GitHub URL. Please use: https://github.com/owner/repo")
        owner = m.group(1)
        repo = m.group(2)
        return owner, repo

    async def _fetch_files(self, owner: str, repo: str, user_token: str = "") -> tuple[list, list]:
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "RepoSage/2.0"}
        # User-supplied token (private repos) wins; used only for this request, never stored or logged
        token = user_token or os.getenv("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        raw_headers = {"Authorization": f"Bearer {token}"} if user_token else {}

        async with httpx.AsyncClient(timeout=60) as client:
            branch = "main"
            r = None
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

            if not r or r.status_code != 200:
                raise ValueError(
                    f"Cannot access {owner}/{repo}. If it's private, add a GitHub token "
                    "(Settings > Developer settings > Fine-grained tokens, read-only 'Contents')."
                )

            tree = r.json().get("tree", [])
            if not tree:
                raise ValueError(f"Repository {owner}/{repo} appears to be empty.")

            # Build full file list for file explorer feature
            all_files = []
            scored = []
            for item in tree:
                if item["type"] != "blob":
                    continue
                path = item["path"]
                ext = os.path.splitext(path)[1].lower()
                size = item.get("size", 0)

                all_files.append({"path": path, "size": size, "ext": ext})

                if ext not in ALLOWED_EXT:
                    continue
                if size > 40_000 or size == 0:
                    continue
                if any(skip in path for skip in SKIP_PATTERNS):
                    continue

                depth = path.count("/")
                is_important = any(
                    name in path.lower()
                    for name in ["main", "app", "index", "core", "api", "server",
                                 "readme", "config", "router", "model", "auth",
                                 "handler", "controller", "service", "util"]
                )
                score = depth * 10 - (5 if is_important else 0)
                scored.append((score, path))

            scored.sort()
            paths = [p for _, p in scored[:20]]

            if not paths:
                raise ValueError("No indexable source files found.")

            tasks = [
                client.get(
                    f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{p}",
                    headers=raw_headers,
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
                    results.append({"path": path, "content": resp.text[:6000]})

            return results, all_files

    def _chunk(self, path: str, content: str) -> list:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".py":
            parts = re.split(r"\n(?=(?:def |class |async def ))", content)
        elif ext in {".js", ".ts", ".jsx", ".tsx", ".vue"}:
            parts = re.split(r"\n(?=(?:function |class |export |const \w+ = ))", content)
        elif ext in {".md", ".txt"}:
            parts = re.split(r"\n(?=#{1,3} )", content)
        else:
            parts = [content[i:i + 900] for i in range(0, len(content), 600)]

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

    async def _generate_summary(self, files: list) -> str:
        # Use top 5 most important files for summary
        important = sorted(
            files,
            key=lambda f: (
                0 if any(n in f["path"].lower() for n in ["readme", "main", "app", "index"]) else 1,
                f["path"].count("/")
            )
        )[:5]
        context = "\n\n---\n\n".join(
            f"# {f['path']}\n{f['content'][:1500]}" for f in important
        )
        try:
            resp = self.ai.messages.create(
                model="claude-haiku-4-5",
                max_tokens=600,
                messages=[{"role": "user", "content": SUMMARY_PROMPT.format(context=context)}]
            )
            return resp.content[0].text
        except Exception:
            return ""

    async def index_repository(self, github_url: str, user_token: str = "", repo_id: str = "") -> dict:
        owner, repo = self._parse_url(github_url)
        repo_id = repo_id or f"{owner}/{repo}"

        files, all_files = await self._fetch_files(owner, repo, user_token)
        if not files:
            raise ValueError("Could not fetch any files. Make sure it's a public repo.")

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
        self.file_lists[repo_id] = all_files

        # Generate summary in background (non-blocking)
        summary = await self._generate_summary(files)
        self.summaries[repo_id] = summary

        return {
            "repo_id": repo_id,
            "file_count": len(files),
            "chunk_count": len(docs),
            "total_files": len(all_files),
            "summary": summary,
        }

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
            if scores[i] > 0:
                context_parts.append(index["docs"][i])
                if index["paths"][i] not in seen:
                    sources.append(index["paths"][i])
                    seen.add(index["paths"][i])

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
            yield {"type": "error", "content": "Rate limit hit. Please wait a moment."}
            return
        except Exception as e:
            yield {"type": "error", "content": f"AI error: {str(e)}"}
            return

        yield {"type": "sources", "content": sources}