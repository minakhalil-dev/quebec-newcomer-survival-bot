# Quebec Newcomer Survival Bot

New in Quebec? Everything is in French, and nobody hands you a manual.

This bot answers newcomer questions in English or French — health card,
driver's licence, renting, taxes, free French classes, childcare,
finding a doctor — from short plain-language summaries, with sources
you can check. If it doesn't know, it says so instead of guessing.

## How it works

Three steps, that's it:

1. **You ask** — in English or French, in the chat.
2. **It looks it up** — it finds the matching pages in its own docs.
3. **It answers with sources** — a short answer, each point linked to
   the doc and the official page behind it. If nothing matches, it
   says "I don't know."

No database, no hosted app. One Python file, a few docs, and a
simple search.

## Try it

```bash
pip install -r requirements.txt
python rag.py
```

Then just ask. Type `quit` to stop.

```bash
python rag.py ask "How do I exchange my foreign driver's licence in Quebec?"
python rag.py ask "Comment s'inscrire à un cours de francisation ?"
python rag.py ask "How do I renew my American passport in Montreal?"  # should say it doesn't know
```

No API key needed: by default it quotes the most relevant sentences
from the docs, with the official page for each source underneath.
Set a Gemini key and it writes a fuller answer from those same
retrieved pages instead.

## A learning project, not a service

I built this to learn how RAG actually works, end to end, on a real
problem — and to keep it small enough to read in one sitting. The
docs are paraphrased summaries, not official pages, so always check
the official source before acting.

Want the details? [How it works, in detail](docs/how-it-works.md) —
the chunking, the retrieval, the eval results, and what's not
measured yet.

> Not legal, tax, medical, or immigration advice. Not affiliated
> with the SAAQ, RAMQ, or any public body. See
> [DISCLAIMER.md](DISCLAIMER.md).
