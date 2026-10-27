#!/usr/bin/env python3
"""Legacy Gradio demo (kept for reference): run with ``python app_gradio.py`` and open http://127.0.0.1:7860."""
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr
from voiceops.agent import Agent
from voiceops.kb import Retriever
from voiceops.pipeline import voice_turn
from voiceops.speech import Speech

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KB, SPEECH = Retriever(), Speech()

CSS = """
:root {
  --ink: #f7f1ff;
  --muted: #b7a9d1;
  --line: rgba(190, 163, 255, .2);
  --canvas: #100b1d;
  --violet: #a855f7;
  --pink: #fb4bba;
  --cyan: #35d9ff;
}
body, .gradio-container {
  background:
    radial-gradient(ellipse at 12% 5%, rgba(168, 85, 247, .2), transparent 38%),
    radial-gradient(ellipse at 90% 25%, rgba(251, 75, 186, .14), transparent 34%),
    var(--canvas) !important;
  color: var(--ink) !important;
  font-family: 'Segoe UI', 'Trebuchet MS', sans-serif !important;
}
.gradio-container {
  max-width: 920px !important;
  min-height: 100vh;
  margin: 0 auto !important;
  padding: 16px 22px 20px !important;
  color-scheme: dark;
  --background-fill-primary: #1b142a;
  --background-fill-secondary: #241a37;
  --body-text-color: #f7f1ff;
  --body-text-color-subdued: #b7a9d1;
  --block-background-fill: #1b142a;
  --border-color-primary: rgba(190, 163, 255, .24);
  --button-primary-background-fill: linear-gradient(110deg, #a855f7, #fb4bba);
  --button-primary-background-fill-hover: linear-gradient(110deg, #bd78ff, #ff70c9);
  --button-primary-text-color: #fff;
}
#app {
  min-height: calc(100vh - 36px);
  gap: 12px !important;
  position: relative;
}
#topbar {
  min-height: 54px;
  align-items: center;
  border-bottom: 1px solid var(--line);
  padding: 0 2px 10px;
}
#brand { display: flex; align-items: center; gap: 10px; }
.brand-mark {
  display: grid; place-items: center; width: 36px; height: 36px;
  border-radius: 13px; color: white; font: 800 18px 'Segoe UI', sans-serif;
  background: linear-gradient(145deg, var(--cyan), var(--violet) 52%, var(--pink));
  box-shadow: 0 0 24px rgba(168, 85, 247, .42);
}
.brand-name { font-size: 15px; font-weight: 750; color: #fff; }
.brand-status { margin-top: 2px; color: var(--muted); font-size: 11px; }
#new-conversation {
  min-height: 38px;
  border-radius: 11px !important;
  border: 1px solid rgba(190, 163, 255, .3) !important;
  background: rgba(168, 85, 247, .1) !important;
  color: var(--ink) !important;
  transition: background .2s ease, border-color .2s ease, transform .2s ease;
}
#new-conversation:hover {
  background: rgba(168, 85, 247, .22) !important;
  border-color: rgba(251, 75, 186, .65) !important;
  transform: translateY(-1px);
}
#chat-history {
  flex: 1;
  min-height: 420px;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}
#chat-history .wrap, #chat-history .bubble-wrap {
  background: transparent !important;
}
#chat-history .placeholder {
  color: var(--muted) !important;
  font-size: 14px !important;
}
#chat-history .message {
  border: 1px solid rgba(190, 163, 255, .16) !important;
  border-radius: 18px !important;
  box-shadow: 0 8px 24px rgba(0, 0, 0, .14);
}
#chat-history .message.user {
  background: linear-gradient(135deg, #6d36b5, #a33d9b) !important;
  color: #fff !important;
}
#chat-history .message.bot {
  background: rgba(35, 26, 54, .92) !important;
  color: var(--ink) !important;
}
#composer {
  gap: 10px !important;
  padding: 12px;
  border: 1px solid rgba(190, 163, 255, .28);
  border-radius: 18px;
  background: linear-gradient(135deg, rgba(35, 26, 54, .96), rgba(27, 20, 42, .96));
  box-shadow: 0 12px 36px rgba(0, 0, 0, .28), 0 0 24px rgba(168, 85, 247, .09);
}
#message-box {
  border: 0 !important;
  box-shadow: none !important;
  background: transparent !important;
}
#message-box textarea {
  font-size: 14px !important;
  color: var(--ink) !important;
  background: transparent !important;
}
#message-box textarea::placeholder { color: #9d8bb9 !important; }
#send {
  min-width: 88px;
  border: 0 !important;
  border-radius: 12px !important;
  background: linear-gradient(110deg, var(--violet), var(--pink)) !important;
  color: #fff !important;
  font-weight: 700 !important;
  box-shadow: 0 5px 18px rgba(201, 70, 207, .28);
  transition: filter .2s ease, transform .2s ease, box-shadow .2s ease;
}
#send:hover {
  filter: brightness(1.13);
  transform: translateY(-1px);
  box-shadow: 0 7px 22px rgba(201, 70, 207, .42);
}
#audio-input {
  border: 1px solid rgba(190, 163, 255, .22) !important;
  border-radius: 12px !important;
  box-shadow: none !important;
  background: rgba(16, 11, 29, .5) !important;
}
#voice-reply {
  border: 1px solid rgba(190, 163, 255, .22) !important;
  border-radius: 12px !important;
  box-shadow: none !important;
  background: rgba(27, 20, 42, .9) !important;
}
#usage-note { color: var(--muted); font-size: 11px; text-align: center; }
button:focus-visible, textarea:focus-visible {
  outline: 2px solid var(--cyan) !important;
  outline-offset: 2px;
}
footer { display: none !important; }
@media (max-width: 640px) {
  .gradio-container { padding: 10px 12px 14px !important; }
  #app { min-height: calc(100vh - 24px); }
  #chat-history { min-height: 340px; }
  #composer { padding: 9px; }
}
"""


def new_agent():
    return Agent(retriever=KB)


def _append_turn(chat, user_message, reply):
    return (chat or []) + [
        {"role": "user", "content": user_message or "(unclear audio)"},
        {"role": "assistant", "content": reply},
    ]


def audio_turn(audio, agent, chat):
    if audio is None:
        return agent, chat, None, ""
    try:
        with tempfile.NamedTemporaryFile(prefix="voiceops-reply-", suffix=".wav", delete=False) as output:
            output_path = output.name
        heard, reply, _ = voice_turn(audio, agent, SPEECH, output_path)
    except Exception as exc:
        logger.exception("Could not process an audio message")
        raise gr.Error("Couldn't process that audio. Check the connection and try again, or send a text message.") from exc
    return agent, _append_turn(chat, heard, reply.text), gr.update(value=output_path, visible=True), ""


def text_turn(message, agent, chat):
    message = (message or "").strip()
    if not message:
        return agent, chat, "", gr.update(value=None, visible=False)
    try:
        reply = agent.respond(message)
    except Exception as exc:
        logger.exception("Could not process a text message")
        raise gr.Error("Couldn't send that message. Please try again.") from exc
    return agent, _append_turn(chat, message, reply.text), "", gr.update(value=None, visible=False)


with gr.Blocks(title="ZentroMart Support") as demo:
    with gr.Column(elem_id="app"):
        with gr.Row(elem_id="topbar"):
            gr.HTML(
                """
                <div id="brand">
                  <div class="brand-mark">Z</div>
                  <div>
                    <div class="brand-name">ZentroMart Support</div>
                    <div class="brand-status">Here to help with orders, returns, and more</div>
                  </div>
                </div>
                """
            )
            new_call = gr.Button("New chat", variant="secondary", elem_id="new-conversation")

        st = gr.State(new_agent)
        chat = gr.Chatbot(
            height=560,
            show_label=False,
            elem_id="chat-history",
            placeholder="Hi! How can I help you today?",
        )
        speak = gr.Audio(
            label="Voice reply",
            show_label=False,
            autoplay=True,
            visible=False,
            elem_id="voice-reply",
        )
        with gr.Column(elem_id="composer"):
            with gr.Row():
                message = gr.Textbox(
                    placeholder="Type your message...",
                    show_label=False,
                    lines=1,
                    max_lines=4,
                    scale=1,
                    elem_id="message-box",
                )
                send = gr.Button("Send", variant="primary", elem_id="send")
            mic = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                recording=False,
                label="Record or upload a voice message",
                elem_id="audio-input",
            )
        gr.HTML(
            '<div id="usage-note">Click Record and allow microphone access when prompted, or type a message/upload audio. '
            'If the device is missing, check browser microphone permissions and reload this page.</div>'
        )

        send.click(text_turn, [message, st, chat], [st, chat, message, speak])
        message.submit(text_turn, [message, st, chat], [st, chat, message, speak])
        mic.stop_recording(audio_turn, [mic, st, chat], [st, chat, speak, message])
        mic.upload(audio_turn, [mic, st, chat], [st, chat, speak, message])
        new_call.click(lambda: (new_agent(), [], None, ""), None, [st, chat, speak, message])


if __name__ == "__main__":
    demo.launch(
        css=CSS,
        theme=gr.themes.Soft(
            primary_hue="teal",
            secondary_hue="blue",
            neutral_hue="slate",
        ),
    )
