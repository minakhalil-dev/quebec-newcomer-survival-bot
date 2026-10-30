#!/usr/bin/env python3
"""
Quebec Newcomer Survival Bot — RAG over public Quebec newcomer docs.

What it does
------------
1. Loads the Markdown docs in ``docs/`` (paraphrased, bilingual summaries of
   public SAAQ / RAMQ / housing / tax / francisation / childcare information
   — NOT verbatim official text, and NOT legal or medical advice).
2. Splits them into chunks of ~512 tokens (approximated as words here;
   see ``chunk_text`` for the approximation and the tiktoken stretch goal).
3. Embeds chunks with TF-IDF by default (zero extra dependencies beyond
   numpy), or with a multilingual sentence-transformer
   (``paraphrase-multilingual-MiniLM-L12-v2``) via ``--embedder sbert``.
4. On ``ask``, retrieves the top-k chunks, and either:
   - returns an *extractive* cited answer (default, no API key needed), or
   - asks Gemini to write a cited answer from the retrieved chunks only,
     when ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` is set.
5. Abstains — "I don't know" — when the best retrieval score is below the
   threshold, instead of hallucinating an answer the docs don't support.

Usage
-----
    pip install -r requirements.txt
    python rag.py                        # interactive chat — the simplest way
    python rag.py chat                   # same, explicit
    python rag.py ask "How do I exchange my foreign driver's licence in Quebec?"
    python rag.py ask "Comment échanger mon permis de conduire ?" --top-k 3
    python rag.py ask "How do I renew a US passport?" --format markdown
    # `index` is optional: ask/chat build the index if it is missing.
    python rag.py index

    # Multilingual embeddings (optional, heavier install):
    pip install sentence-transformers
    python rag.py ask "How do I get a RAMQ card?" --embedder sbert

Design notes for interviews
---------------------------
- Why TF-IDF first? It makes the baseline measurable and runnable anywhere.
  The eval table in README.md compares it against multilingual embeddings,
  which is exactly the cross-lingual problem here: questions in English,
  sources largely in French.
- Why citations like ``[saaq-permis-de-conduire.md#chunk-1]``? A newcomer
  bot that cannot show its source should not be trusted with a lease,
  a licence, or a health card.
- Why abstain? A wrong confident answer about a deadline is worse than
  no answer. The abstention cases in ``eval/questions.json`` test this.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
DOCS_DIR = ROOT / "docs"
INDEX_DIR = ROOT / ".index"

# Official page for each doc, so answers can link out instead of only
# citing a local filename. Kept in sync with docs/sources.md.
# Maps doc filename -> (human-readable name, official URL).
SOURCE_URLS = {
    "saaq-permis-de-conduire.md": ("SAAQ", "https://saaq.gouv.qc.ca"),
    "ramq-assurance-maladie.md": ("RAMQ", "https://www.ramq.gouv.qc.ca"),
    "logement-droits-locataire.md": (
        "Tribunal administratif du logement",
        "https://www.tal.gouv.qc.ca",
    ),
    "revenus-quebec-impots.md": ("Revenu Québec", "https://www.revenuquebec.ca"),
    "francisation-cours-francais.md": (
        "Francisation Québec",
        "https://www.quebec.ca/education/apprendre-francais",
    ),
    "garderie-cpe.md": ("Place 0-5", "https://www.place0-5.com"),
    "sante-acces-medecin.md": ("Info-Santé / GAMF", "https://www.quebec.ca/sante"),
}


def source_info(source: str) -> tuple[str, str | None]:
    """Return (display name, official URL or None) for a doc filename."""
    return SOURCE_URLS.get(source, (source, None))


def relevant_sources(hits: list[dict], threshold: float) -> list[dict]:
    """Hits that actually cleared the threshold, with clickable metadata."""
    out = []
    for h in hits:
        if h["score"] < threshold:
            continue
        name, url = source_info(h["source"])
        out.append({"id": h["id"], "source": h["source"], "name": name, "url": url})
    return out

# Small EN + FR stopword list. This is a baseline, not a linguistic claim.
STOPWORDS = set(
    """
    a an the and or of to in on for with is are was were be been it its this that
    le la les de des du un une et ou à au aux en dans pour sur est sont avec
    ce cette ces il elle nous vous ils elles pas plus que qui quoi comment
    how do does did what when where which who can i my me je mon ma mes
    """.split()
)


# --------------------------------------------------------------------------
# Loading + chunking
# --------------------------------------------------------------------------

def load_docs(docs_dir: Path = DOCS_DIR) -> list[dict]:
    """Load every ``*.md`` in ``docs/``. Returns [{source, text}]."""
    docs = []
    for path in sorted(docs_dir.glob("*.md")):
        if path.name.lower() == "sources.md":
            continue
        docs.append({"source": path.name, "text": path.read_text(encoding="utf-8")})
    if not docs:
        raise SystemExit(f"No docs found in {docs_dir}. Did you clone the full repo?")
    return docs


def _approx_words_for_tokens(n_tokens: int) -> int:
    # Rough rule of thumb: ~1.33 tokens per word for English/French prose.
    # A production version would use tiktoken / a HF tokenizer and chunk
    # on exact token counts; the eval in eval.py measures whether the
    # approximation hurts retrieval, which is the honest comparison.
    return max(20, int(n_tokens / 1.33))


def chunk_text(text: str, source: str, chunk_tokens: int = 512, overlap_tokens: int = 64) -> list[dict]:
    """Split text into overlapping word windows.

    Overlap exists so a fact straddling a boundary is not lost. Chunk ids
    are stable and citable: ``{source}#chunk-{n}``.
    """
    chunk_words = _approx_words_for_tokens(chunk_tokens)
    overlap_words = _approx_words_for_tokens(overlap_tokens)
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_words - overlap_words)
    chunks = []
    for i, start in enumerate(range(0, len(words), step)):
        window = words[start : start + chunk_words]
        if not window:
            break
        chunks.append(
            {
                "id": f"{source}#chunk-{i}",
                "source": source,
                "text": " ".join(window),
            }
        )
        if start + chunk_words >= len(words):
            break
    return chunks


def build_chunks(docs: list[dict], chunk_tokens: int = 512, overlap_tokens: int = 64) -> list[dict]:
    chunks: list[dict] = []
    for doc in docs:
        chunks.extend(chunk_text(doc["text"], doc["source"], chunk_tokens, overlap_tokens))
    return chunks


# --------------------------------------------------------------------------
# TF-IDF embedder (numpy only — no sklearn needed)
# --------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    # Strip accents so "Québec" and "Quebec" match, lowercase, keep words.
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    tokens = re.findall(r"[a-z0-9]+", text)
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


class TfidfEmbedder:
    name = "tfidf"

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None

    def fit(self, texts: list[str]) -> None:
        doc_freq: Counter = Counter()
        for text in texts:
            for tok in set(_tokenize(text)):
                doc_freq[tok] += 1
        self.vocab = {tok: i for i, tok in enumerate(sorted(doc_freq))}
        n_docs = max(1, len(texts))
        self.idf = np.array(
            [math.log((1 + n_docs) / (1 + doc_freq[tok])) + 1.0 for tok in sorted(doc_freq)],
            dtype=np.float32,
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.idf is None:
            raise RuntimeError("Call fit() before encode().")
        mat = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for row, text in enumerate(texts):
            counts = Counter(_tokenize(text))
            for tok, count in counts.items():
                if tok in self.vocab:
                    # log-scaled TF x IDF
                    mat[row, self.vocab[tok]] = (1 + math.log(count)) * self.idf[self.vocab[tok]]
            norm = np.linalg.norm(mat[row])
            if norm > 0:
                mat[row] /= norm
        return mat


class SbertEmbedder:
    """Multilingual embeddings. Optional — only used with --embedder sbert."""

    name = "sbert"

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - install-time path
            raise SystemExit(
                "sentence-transformers is not installed. Run:\n"
                "    pip install sentence-transformers\n"
                "and retry with --embedder sbert."
            ) from exc
        self.model = SentenceTransformer(model_name)

    def fit(self, texts: list[str]) -> None:  # noqa: ARG002 - API parity
        return None

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


def get_embedder(name: str):
    if name == "tfidf":
        return TfidfEmbedder()
    if name == "sbert":
        return SbertEmbedder()
    raise SystemExit(f"Unknown embedder: {name!r} (use 'tfidf' or 'sbert')")


# --------------------------------------------------------------------------
# Index + retrieval
# --------------------------------------------------------------------------

class Index:
    def __init__(self, chunks: list[dict], embedder, vectors: np.ndarray):
        self.chunks = chunks
        self.embedder = embedder
        self.vectors = vectors

    def save(self, index_dir: Path = INDEX_DIR) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        (index_dir / "chunks.json").write_text(
            json.dumps(self.chunks, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        np.save(index_dir / "vectors.npy", self.vectors)
        meta = {"embedder": self.embedder.name}
        if isinstance(self.embedder, TfidfEmbedder):
            meta["vocab"] = self.embedder.vocab
            meta["idf"] = self.embedder.idf.tolist() if self.embedder.idf is not None else []
        (index_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, embedder_name: str, index_dir: Path = INDEX_DIR):
        chunks_path, vectors_path, meta_path = (
            index_dir / "chunks.json",
            index_dir / "vectors.npy",
            index_dir / "meta.json",
        )
        if not (chunks_path.exists() and vectors_path.exists()):
            return None
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        vectors = np.load(vectors_path)
        embedder = get_embedder(embedder_name)
        if embedder_name == "tfidf" and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            embedder.vocab = meta.get("vocab", {})
            idf = meta.get("idf", [])
            embedder.idf = np.array(idf, dtype=np.float32) if idf else None
        return cls(chunks, embedder, vectors)


def build_index(chunk_tokens: int = 512, overlap_tokens: int = 64, embedder_name: str = "tfidf") -> Index:
    docs = load_docs()
    chunks = build_chunks(docs, chunk_tokens, overlap_tokens)
    embedder = get_embedder(embedder_name)
    embedder.fit([c["text"] for c in chunks])
    vectors = embedder.encode([c["text"] for c in chunks])
    return Index(chunks, embedder, vectors)


def retrieve(index: Index, query: str, top_k: int = 5) -> list[dict]:
    """Return top-k chunks with cosine scores, highest first."""
    q_vec = index.embedder.encode([query])[0]
    scores = index.vectors @ q_vec  # vectors are L2-normalised
    order = np.argsort(-scores)[:top_k]
    return [
        {"score": float(scores[i]), "id": index.chunks[i]["id"],
         "source": index.chunks[i]["source"], "text": index.chunks[i]["text"]}
        for i in order
    ]


# --------------------------------------------------------------------------
# Answering
# --------------------------------------------------------------------------

ABSTAIN_MESSAGE_EN = (
    "I don't know from the docs I have — and I won't guess on a licence, "
    "health card, lease, or tax question. Check the official source in "
    "docs/sources.md, or ask a question covered by the docs in this repo."
)
ABSTAIN_MESSAGE_FR = (
    "Je ne sais pas à partir des documents dont je dispose, et je ne vais "
    "pas deviner pour un permis, une carte d'assurance maladie, un bail ou "
    "une question fiscale. Vérifiez la source officielle dans docs/sources.md."
)


def _is_french(query: str) -> bool:
    # NOTE: do not use _tokenize() here — it strips stopwords, and most
    # French markers (comment, les, des, est, ...) are stopwords, so the
    # old version returned False for almost every French question.
    # Check the raw words instead.
    text = unicodedata.normalize("NFKD", query.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    words = set(re.findall(r"[a-z]+", text))
    french_markers = {
        "comment", "pourquoi", "puis", "est", "ce", "les", "des", "du",
        "un", "une", "le", "la", "pour", "dans", "avec", "qui", "que",
        "sont", "inscrire", "cours", "francisation", "permis", "conduire",
        "bail", "locataire", "sante", "impots", "garderie",
    }
    return bool(french_markers & words)


def _clean_text(text: str) -> str:
    """Strip Markdown markers and collapse whitespace for display.

    The old extractive answer dumped the first 600 characters of a chunk
    verbatim, which meant headings, half-sentences, and mid-word cuts.
    Cleaning + sentence selection below fixes that without inventing
    any new facts: every sentence shown still comes from the docs.
    """
    # Chunk text is word-joined, so headings are mid-string, not at a
    # line start — strip them everywhere, not just at line starts.
    text = re.sub(r"#{1,6}\s*", "", text)
    text = text.replace("**", "").replace("__", "").replace("`", "")
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)
    return re.sub(r"\s+", " ", text).strip()


def _best_sentences(query: str, text: str, max_sentences: int = 4) -> list[str]:
    """Pick the most query-relevant complete sentences from a chunk.

    Sentences are scored by how many distinct query words they contain,
    then returned in their original order so the answer reads naturally.
    If nothing scores, fall back to the first complete sentences.
    """
    cleaned = _clean_text(text)
    raw_sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()
    ]
    # For an English question, prefer the English summary over the French
    # body: French sentences usually carry several accented characters,
    # English ones (in these docs) carry at most "Québec". Only apply
    # the filter when enough sentences survive it, so a French-only
    # passage is never hidden entirely.
    if not _is_french(query):
        unaccented = [
            s
            for s in raw_sentences
            if sum(1 for c in s if c in "àâäéèêëîïôöùûüç") < 2
        ]
        if len(unaccented) >= 2:
            raw_sentences = unaccented
    sentences = []
    for s in raw_sentences:
        s = s.strip()
        # Drop doc metadata, not answers: keyword lines and source headers.
        if not s or "mots-clés" in s.lower() or "keywords:" in s.lower():
            continue
        if "source officielle" in s.lower() or "sources.md" in s.lower():
            continue
        if s.lower().startswith("english summary "):
            s = s[len("english summary ") :].strip()
            s = s[:1].upper() + s[1:] if s else s
        if len(s) >= 40:
            sentences.append(s)
    if not sentences:
        return [cleaned[:400]] if cleaned else []
    query_tokens = set(_tokenize(query))
    french_query = _is_french(query)
    scored = []
    for i, sentence in enumerate(sentences):
        overlap = len(query_tokens & set(_tokenize(sentence)))
        accents = sum(1 for c in sentence if c in "àâäéèêëîïôöùûüç")
        # Tie-break by language: an English question should prefer the
        # English summary over a French sentence that only matches
        # "Quebec", and vice versa for a French question.
        scored.append((overlap, -accents if french_query else accents, i, sentence))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    scored = [(overlap, i, sentence) for overlap, _, i, sentence in scored]
    picked = [item for item in scored if item[0] > 0][:max_sentences]
    if not picked:
        picked = scored[:2]
    picked.sort(key=lambda item: item[1])
    return [sentence for _, _, sentence in picked]


def _gemini_answer(query: str, hits: list[dict]) -> str | None:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None
    try:
        import google.generativeai as genai
    except ImportError:
        return None
    genai.configure(api_key=api_key)
    context = "\n\n".join(f"[{h['id']}]\n{h['text']}" for h in hits)
    prompt = (
        "Answer ONLY from the context below. Cite every factual claim as "
        "[source#chunk-n] using the ids given. If the context does not cover "
        "the question, say 'I don't know from these docs.' Do not add facts "
        "from outside the context.\n\n"
        "Write for a newcomer, in plain, simple language: start with a "
        "1-2 sentence summary, then short bullet points for the steps or "
        "key facts. Keep paragraphs short and do not dump long blocks of "
        "text.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    model = genai.GenerativeModel("gemini-2.0-flash")
    return model.generate_content(prompt).text.strip()


def answer_question(
    index: Index, query: str, top_k: int = 5, threshold: float = 0.12
) -> dict:
    """Retrieve + answer. ``abstained=True`` when nothing scores high enough."""
    hits = retrieve(index, query, top_k=top_k)
    if not hits or hits[0]["score"] < threshold:
        return {
            "query": query,
            "abstained": True,
            "answer": ABSTAIN_MESSAGE_FR if _is_french(query) else ABSTAIN_MESSAGE_EN,
            "hits": hits,
            "sources": [],
        }
    generated = _gemini_answer(query, hits)
    if generated:
        answer = generated
    else:
        # Extractive fallback: no API key, no hallucination. Instead of
        # dumping the start of each chunk (which cut sentences mid-word
        # and buried the English summary behind a French intro), show the
        # most query-relevant complete sentences from each hit, as bullets.
        # Only hits that cleared the threshold count — a 0.02-score chunk
        # is noise, not evidence.
        bullets: list[str] = []
        for h in [h for h in hits if h["score"] >= threshold][:3]:
            name, _url = source_info(h["source"])
            for sentence in _best_sentences(query, h["text"]):
                bullets.append(f"- {sentence} ({name})")
        if not bullets:
            bullets = ["- No readable passage cleared the threshold."]
        answer = "What the docs say:\n\n" + "\n".join(bullets)
    return {
        "query": query,
        "abstained": False,
        "answer": answer,
        "hits": hits,
        "sources": relevant_sources(hits, threshold),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _print_result(result: dict, fmt: str) -> None:
    """One display path for ask and chat, so both stay readable.

    Text mode prints bare official URLs (clickable in most terminals).
    Markdown mode prints proper [name](url) links, for GitHub, a
    Markdown viewer, or pasting into a doc. Only sources that cleared
    the threshold are listed — low-score hits were noise, not evidence.
    """
    if fmt == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(result["answer"])
    if result["abstained"]:
        print("\nNote: this is an abstention, not an answer. Check the official source in docs/sources.md.")
        return
    sources = result.get("sources") or []
    if not sources:
        return
    if fmt == "markdown":
        print("\n## Sources\n")
        for s in sources:
            if s["url"]:
                print(f"- [{s['name']}]({s['url']}) — {s['id']}")
            else:
                print(f"- {s['name']} — {s['id']}")
    else:
        print("\nSources (official pages — click to check):")
        for s in sources:
            if s["url"]:
                print(f"  - {s['name']} — {s['url']} ({s['id']})")
            else:
                print(f"  - {s['name']} ({s['id']})")


def _load_or_build(embedder: str, chunk_tokens: int) -> Index:
    # Prefer a saved index, otherwise build in memory. This is what makes
    # `ask` work without a separate `index` step first.
    index = Index.load(embedder)
    if index is None:
        index = build_index(chunk_tokens, 64, embedder)
    return index


def _run_chat(embedder: str, chunk_tokens: int, top_k: int, threshold: float, fmt: str) -> int:
    """Interactive loop: one process, no re-indexing, no shell quoting.

    Ask as many questions as you like; type quit, exit, or Ctrl-C/D to stop.
    """
    index = _load_or_build(embedder, chunk_tokens)
    print("Quebec Newcomer Survival Bot — ask in English or French.")
    print("Type 'quit' to stop.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return 0
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Bye!")
            return 0
        result = answer_question(index, question, top_k=top_k, threshold=threshold)
        print()
        _print_result(result, fmt)
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quebec Newcomer Survival Bot (RAG)")
    sub = parser.add_subparsers(dest="command")

    p_index = sub.add_parser("index", help="Build and save the vector index (optional — ask/chat build it if missing)")
    p_index.add_argument("--embedder", default="tfidf", choices=["tfidf", "sbert"])
    p_index.add_argument("--chunk-tokens", type=int, default=512)
    p_index.add_argument("--overlap-tokens", type=int, default=64)

    p_ask = sub.add_parser("ask", help="Ask one question")
    p_ask.add_argument("question", nargs="?", default=None, help="If omitted, you will be prompted for it")
    p_ask.add_argument("--embedder", default="tfidf", choices=["tfidf", "sbert"])
    p_ask.add_argument("--chunk-tokens", type=int, default=512)
    p_ask.add_argument("--top-k", type=int, default=5)
    p_ask.add_argument("--threshold", type=float, default=0.12)
    p_ask.add_argument("--format", default="text", choices=["text", "json", "markdown"])

    p_chat = sub.add_parser("chat", help="Interactive mode: ask question after question, no re-typing the command")
    p_chat.add_argument("--embedder", default="tfidf", choices=["tfidf", "sbert"])
    p_chat.add_argument("--chunk-tokens", type=int, default=512)
    p_chat.add_argument("--top-k", type=int, default=5)
    p_chat.add_argument("--threshold", type=float, default=0.12)
    p_chat.add_argument("--format", default="text", choices=["text", "json", "markdown"])

    args = parser.parse_args(argv)

    # No subcommand at all = chat. `python rag.py` should just work.
    if args.command is None:
        return _run_chat("tfidf", 512, 5, 0.12, "text")

    if args.command == "index":
        index = build_index(args.chunk_tokens, args.overlap_tokens, args.embedder)
        index.save()
        print(f"Indexed {len(index.chunks)} chunks with embedder={args.embedder} -> .index/")
        return 0

    if args.command == "chat":
        return _run_chat(args.embedder, args.chunk_tokens, args.top_k, args.threshold, args.format)

    # ask
    question = args.question
    if not question:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            return 0
    index = _load_or_build(args.embedder, args.chunk_tokens)
    result = answer_question(index, question, top_k=args.top_k, threshold=args.threshold)
    _print_result(result, args.format)
    return 0


if __name__ == "__main__":
    sys.exit(main())
