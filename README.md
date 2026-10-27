# VoiceOps 2026: ZentroMart voice support agent

A customer-support agent that answers from ZentroMart's documents, remembers the conversation, verifies callers,
uses the order/ticket tools, and speaks and listens through **Gnani STT/TTS** (the only speech services used).

```
audio -> Gnani STT -> [ analyze (LLM) -> identity/state (code) -> ESC rules (code) -> facts (code)
                        -> retrieve KB (hybrid) -> respond (LLM) -> guards (code) ] -> Gnani TTS -> audio
```

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then fill GNANI_API_KEY and ONE LLM key. Never commit .env
# Extract Capstone_Project_Dataset_FTTT_2026.zip next to this folder (or set DATASET_DIR in .env)
python Capstone_Project_Dataset_FTTT_2026/validate_dataset.py
```
Optional, better Hindi/Hinglish retrieval: `pip install sentence-transformers numpy` (hybrid mode picks it up automatically).

## Run command (what the organizers use)
```bash
./run.sh <folder_of_audio> [out_folder]        # = python run.py --audio-dir <folder> --out out
```
Per file it saves `<name>.json` (transcript, reply text, spoken text, decision, tool calls, retrieved chunks + the ones
used), `<name>.txt`, `<name>_reply.wav`, and a `results.csv`. Files named `PC002_t1.wav, PC002_t2.wav ...` are played as
one conversation; any other name is an independent call. A bad file never stops the batch.

## Evaluate on the practice data
```bash
python evaluate.py step1 --no-cache # prompt-only LLM    -> "Step 1" score for the report (--no-cache = fresh LLM calls)
python evaluate.py step2            # RAG agent          -> "Step 2" score
python evaluate.py step2 --asr      # same on asr_text (speech-like input)
python evaluate.py step3            # 23 conversations / 96 turns (tools, memory, handover)
python -m pytest tests -q           # offline unit + plumbing tests (no keys needed)
python app.py                       # web UI with mic, case file and sources (opens your browser)
```

## How it works (and why)
* **Code does what must be exact.** Business-day refund ETAs, return deadlines (7/10 day electronics rule, festive 30-day
  fashion rule), lateness, overdue-refund detection, Assured-vs-marketplace rule and the Rs 10,000 handover threshold are
  computed in `dates.py` / `facts.py` and handed to the LLM as facts. The LLM never does calendar arithmetic.
* **Identity check is a state machine** (`agent.py`): order ID -> `lookup_order(id)` -> ask for ONE of phone-last-4 / pincode
  -> verify -> max one retry -> locked (policy help only). Order data is only used after `verified=True`; verification values are
  redacted from history, logs and ticket summaries; phone numbers never reach the LLM.
* **Escalation (ESC-1)** is rule-based: human requested, legal, safety, unauthorized activity, `contacts_last_7_days >= 2`, refund >3
  business days past max, damage claim > Rs 10,000. A ticket is created first; the reply claims a handover only if `created=True`.
* **Grounding**: section-level KB chunks keep `[ID]`, status, effective_from; precedence policy > guide > help > FAQ > marketing;
  superseded v3 returns policy only surfaces for electronics delivered before 1 Aug 2026; FAQ lines shadowed by a policy are dropped
  (and KB-00.1 cited). The responder may only use retrieved text + computed facts, and says "I don't know" + offers a human otherwise.
* **Memory**: order, verification, last product(s), pending question (answered right after verification), language.
* **Voice**: STT runs `en-IN` (English + Hinglish) and `hi-IN` in parallel; hypotheses go to the analyzer. Replies are rewritten for the
  ear (dates "27th of October 2026", order IDs in groups of four, "49 rupees"). TTS always names the model; non-English replies use
  `timbre-v3.0`. SDK 0.8.x rejects `hi-en` for TTS, so it falls back to `hi-IN`.
* Guards: no claiming cancellations/approvals/credits, no ticket claims without a ticket, AI disclosure on the first reply.

## Layout
`src/voiceops/` agent, kb, facts, dates, speech, templates, prompts | `run.py` batch | `evaluate.py` | `app.py` + `web/` (UI) + `src/voiceops/server.py` | `app_gradio.py` (old demo) | `tests/`

## Known limits
Not verified against a live LLM or live Gnani in the build environment: run `evaluate.py` and a few recordings with your keys.
Languages beyond en/hi/hi-en rely on the LLM for scripted lines. Mock-tool identity check is a simulation, not real authentication.

## The web UI (`python app.py`)
Talk or type. Tap the mic, speak, and it stops by itself when you pause (tap again to stop early, Esc to cancel).
The reply text appears as soon as the agent has it; the voice follows a moment later. The **Case file** panel
shows what the agent did for the selected reply: the order and identity-check state, which policy sections it
retrieved and used (click to read them), the tools it called, how it understood a follow-up, any handover ticket, and
the time each step took. *Hands-free* starts listening again after each spoken reply; *Read aloud* also speaks replies
to typed messages. The browser only records audio; all speech-to-text and text-to-speech goes through Gnani.
Microphones need `localhost` or https, so keep `--host 127.0.0.1` unless you add https.

## Speed: what was changed and why
Local compute per turn is ~2 ms; the time is network round trips, so the work targets those:
| Change | Effect |
|---|---|
| Greeting / "I need help with an order" opener and an order-ID-only turn are answered by templates | on the 23 practice conversations, LLM calls drop 100 -> 56; those turns need no LLM at all. Any extra word falls back to the LLM |
| One shared keep-alive HTTP connection pool for LLM calls | the 2-3 calls in a turn reuse one TLS connection |
| STT: use the first language's answer, wait at most `STT_GRACE_S` (1 s) for the others | a slow second language no longer holds the turn |
| Reply text is streamed to the UI before the audio is generated | the customer reads the answer while the voice is still being made |
| Disk cache (`.cache/`) for LLM answers and TTS audio; scripted voice lines are pre-generated at start | repeat questions, repeat eval runs and the intro/ID prompts are instant |
| Bad key / bad request fails at once instead of retrying 3x with sleeps | errors show immediately |
Speed knobs are listed in `.env.example` (`LLM_ANALYZER_MODEL` and `LLM_REASONING_EFFORT` are the biggest remaining levers).
Only validated JSON is cached, and the cache key contains the whole prompt, so changing a prompt or the code never replays stale output.
Use `--no-cache` when you want fresh numbers for the report. Delete `.cache/` any time.
