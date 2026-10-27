# VoiceOps 2026 report (max 4 pages; export to report.pdf)

**Team:** ...  **LLM:** ...  **Embeddings:** BM25 + (multilingual-e5-small | none)

## 1. What I built
- Step 1 prompt-only assistant (system prompt in `prompts.py: STEP1_SYSTEM`)
- Step 2 grounded RAG (section chunks, hybrid retrieval, precedence, supersession)
- Step 3 conversation (state machine, memory, tools, ESC rules)
- Step 4 voice (Gnani STT/TTS, ear-friendly text)

## 2. Step 1 vs Step 2  (paste from `python evaluate.py step1` / `step2`)
| Metric | Step 1 | Step 2 |
|---|---|---|
| Correct (decision + facts + no forbidden) | _/52 | _/52 |
| Mean evidence recall | n/a | _% |
Step 3 conversations: _/96 turns correct, tool calls ok _%. ASR variant (step2 --asr): _/52.

## 3. What went wrong / what I'd fix
(Read `eval_results/*.json` failing rows; list 3-5 concrete failures and the fix.)

## 4. Bonus
Language switching, ASR number repair, 60-second clip splitting, dense retrieval, mic demo UI.
