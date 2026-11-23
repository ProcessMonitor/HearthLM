"""OpenAI-compatible chat completions endpoint.

    python -m hearthlm.serve --checkpoint out/checkpoints/sft.pt --port 8000

Implemented on `http.server` on purpose: the whole point of this project is that
you can read the stack end to end, and pulling in FastAPI + uvicorn + pydantic to
serve a 5M-parameter model is a lot of unread code. If you need concurrency and
throughput, run vLLM against the exported HuggingFace format instead.

Supported: POST /v1/chat/completions (stream and non-stream), GET /v1/models,
GET /health, and a tiny built-in web page at /.
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

import torch

from ..chat.session import ChatSession, split_thinking
from ..data import Tokenizer
from ..data.chat_template import render_text
from ..model import SamplingParams
from ..model.generate import generate
from ..train.optim import load_model_from_file
from ..train.runtime import pick_device, resolve_dtype


INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>HearthLM</title>
<style>
  body{font:15px/1.5 ui-sans-serif,system-ui,sans-serif;max-width:46rem;margin:3rem auto;padding:0 1rem;color:#222}
  #log{min-height:14rem;white-space:pre-wrap;border:1px solid #ddd;border-radius:8px;padding:1rem;margin-bottom:1rem}
  .u{color:#0a58ca;font-weight:600} .a{color:#146c43;font-weight:600}
  textarea{width:100%;height:5rem;font:inherit;padding:.6rem;box-sizing:border-box;border-radius:8px;border:1px solid #ccc}
  button{margin-top:.5rem;padding:.5rem 1.1rem;font:inherit;border-radius:8px;border:0;background:#222;color:#fff;cursor:pointer}
</style></head><body>
<h3>HearthLM</h3>
<div id="log"></div>
<textarea id="box" placeholder="Ask something, Enter to send"></textarea>
<button onclick="send()">Send</button>
<script>
const log=document.getElementById('log'), box=document.getElementById('box');
box.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();}});
function add(cls,t){const d=document.createElement('div');d.innerHTML='<span class="'+cls+'">'+cls.toUpperCase()+': </span>';d.appendChild(document.createTextNode(t));log.appendChild(d);return d;}
async function send(){
  const text=box.value.trim(); if(!text) return; box.value='';
  add('u',text); const row=document.createElement('div');
  row.innerHTML='<span class="a">ASSISTANT: </span>';
  const span=document.createElement('span'); row.appendChild(span); log.appendChild(row);
  const res=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({messages:[{role:'user',content:text}],stream:true,max_tokens:200})});
  const reader=res.body.getReader(), dec=new TextDecoder(); let buf='';
  while(true){
    const {value,done}=await reader.read(); if(done) break;
    buf+=dec.decode(value,{stream:true});
    const lines=buf.split('\\n'); buf=lines.pop();
    for(const line of lines){
      if(!line.startsWith('data: ')) continue;
      const payload=line.slice(6).trim(); if(payload==='[DONE]') continue;
      try{const j=JSON.parse(payload); const d=j.choices[0].delta||{}; if(d.content) span.textContent+=d.content;}catch(e){}
    }
  }
  log.scrollTop=log.scrollHeight;
}
</script></body></html>"""


class ModelRuntime:
    """Holds the model and serialises access; a single decoder is not thread safe."""

    def __init__(self, checkpoint: str, tokenizer_dir: str, device: str, dtype: str):
        self.device = pick_device(device)
        self.dtype = resolve_dtype(dtype, self.device)
        model, self.config, _ = load_model_from_file(checkpoint, _model_cls(), _config_cls())
        self.model = model.to(self.device, dtype=self.dtype if self.dtype != torch.float32 else None).eval()
        if self.dtype == torch.float32:
            self.model = self.model.float()
        self.tokenizer = Tokenizer(tokenizer_dir)
        import threading

        self._lock = threading.Lock()

    def complete(
        self,
        prompt: str,
        params: SamplingParams,
        *,
        stream: bool = False,
        stats: Optional[dict] = None,
    ):
        ids = self.tokenizer.encode(prompt)
        inp = torch.tensor([ids], device=self.device)
        with self._lock:
            t0 = time.perf_counter()
            out = generate(
                self.model, inp, params,
                eos_token_id=self.config.eos_token_id,
                stop_token_ids=params.stop_token_ids or None,
            )
            elapsed = time.perf_counter() - t0
        new_ids = out[0, len(ids):].tolist()
        if stats is not None:
            stats["prompt_tokens"] = len(ids)
            stats["completion_tokens"] = len(new_ids)
            stats["seconds"] = elapsed
        text = self.tokenizer.decode(new_ids)
        reasoning, answer = split_thinking(text)
        return answer, reasoning, len(ids), len(new_ids)


def _model_cls():
    from ..model.hearth import HearthLM

    return HearthLM


def _config_cls():
    from ..config import HearthConfig

    return HearthConfig


def make_handler(runtime: ModelRuntime):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quieter than the default
            pass

        def _send_json(self, obj: dict, status: int = 200) -> None:
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path in {"/", "/index.html"}:
                body = INDEX_HTML.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/v1/models":
                self._send_json({"object": "list", "data": [
                    {"id": "hearthlm", "object": "model", "owned_by": "local"}
                ]})
            elif self.path == "/health":
                self._send_json({"status": "ok", "device": str(runtime.device)})
            else:
                self._send_json({"error": "not found"}, 404)

        def do_POST(self):  # noqa: N802
            if self.path != "/v1/chat/completions":
                self._send_json({"error": "not found"}, 404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError as exc:
                self._send_json({"error": {"message": f"invalid json: {exc}"}}, 400)
                return

            messages = body.get("messages")
            if not isinstance(messages, list) or not messages:
                self._send_json({"error": {"message": "messages must be a non-empty list"}}, 400)
                return

            tools = body.get("tools")
            thinking = bool(body.get("enable_thinking", False))
            prompt = render_text(
                messages,
                add_generation_prompt=True,
                tools=tools,
                enable_thinking=thinking,
            )
            params = SamplingParams(
                max_new_tokens=int(body.get("max_tokens") or 256),
                temperature=float(body.get("temperature", 0.7)),
                top_p=float(body.get("top_p", 0.9)),
                top_k=int(body.get("top_k", 40)),
                repetition_penalty=float(body.get("repetition_penalty", 1.0)),
                do_sample=float(body.get("temperature", 0.7)) > 0,
            )
            stats: dict = {}
            answer, reasoning, n_prompt, n_out = runtime.complete(prompt, params, stats=stats)

            cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created = int(time.time())
            if body.get("stream"):
                self._stream(cid, created, answer, reasoning)
            else:
                message = {"role": "assistant", "content": answer}
                if reasoning:
                    message["reasoning_content"] = reasoning
                self._send_json({
                    "id": cid,
                    "object": "chat.completion",
                    "created": created,
                    "model": body.get("model", "hearthlm"),
                    "choices": [{
                        "index": 0,
                        "message": message,
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "prompt_tokens": n_prompt,
                        "completion_tokens": n_out,
                        "total_tokens": n_prompt + n_out,
                    },
                })

        def _stream(self, cid: str, created: int, answer: str, reasoning: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            # SSE has no length, so the connection itself delimits the body. Say so
            # explicitly and close after DONE; otherwise clients wait for EOF.
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

            def emit(delta: dict, finish: Optional[str] = None) -> None:
                chunk = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()

            emit({"role": "assistant"})
            if reasoning:
                emit({"reasoning_content": reasoning})
            # Re-chunk on word boundaries; the model already finished generating,
            # so this only shapes the transport, it does not save time.
            buf = ""
            for ch in answer:
                buf += ch
                if ch in " \n.,!?;:":
                    emit({"content": buf})
                    buf = ""
            if buf:
                emit({"content": buf})
            emit({}, finish="stop")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    return Handler


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="HearthLM OpenAI-compatible server")
    p.add_argument("--checkpoint", default="out/checkpoints/sft.pt")
    p.add_argument("--tokenizer", default="assets/tokenizer")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    args = p.parse_args(argv)

    runtime = ModelRuntime(args.checkpoint, args.tokenizer, args.device, args.dtype)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(runtime))
    print(f"serving on http://{args.host}:{args.port}  (device={runtime.device})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        server.shutdown()


if __name__ == "__main__":
    main()