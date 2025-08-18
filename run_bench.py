#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark LiteLLM: SDK vs Proxy
--------------------------------
- Compara latência/TTFT/erros entre:
  (A) LiteLLM SDK (chamando provedores direto)  -> --mode sdk
  (B) LiteLLM Proxy (HTTP OpenAI-compatible)    -> --mode proxy

Cenários:
  --scenario chat    -> não-stream, sem tools
  --scenario stream  -> streaming (mede TTFT)
  --scenario tools   -> não-stream com "tools" no JSON

Saídas:
  reports/results_*.csv  -> linha a linha (em tempo real)
  reports/summary_*.md   -> agregados + custo estimado (pricing.yaml)

Requisitos:
  pip install litellm openai python-dotenv pyyaml
"""

import os, asyncio, time, json, argparse, math, csv
from pathlib import Path
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
import yaml

# SDKs
import litellm
from openai import AsyncOpenAI  # só para falar com o PROXY (OpenAI-compatible)

load_dotenv()

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# -------------------------
# Util: preços e custo
# -------------------------
def load_pricing() -> Dict[str, Dict[str, float]]:
    """
    pricing.yaml esperado (USD por 1K tokens):
      openai/gpt-4o-mini:
        input: 0.0006
        output: 0.0024
    """
    with open("pricing.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def estimate_cost(model: str, input_tokens: int, output_tokens: int, pricing: Dict[str, Dict[str, float]]) -> float:
    table = pricing.get(model, None)
    if not table:
        return 0.0
    return (input_tokens/1000.0)*float(table["input"]) + (output_tokens/1000.0)*float(table["output"])

# -------------------------
# Modelos e clientes
# -------------------------
def get_model(mode: str) -> str:
    """
    - Para SDK (LiteLLM): LITELLM_SDK_MODEL (ex.: 'openai/gpt-4o-mini' ou 'anthropic/claude-3-5-sonnet')
    - Para Proxy:         PROXY_MODEL       (ex.: 'openai/gpt-4o-mini' ou um alias do proxy)
    """
    if mode == "sdk":
        return os.getenv("LITELLM_SDK_MODEL")
    elif mode == "proxy":
        return os.getenv("PROXY_MODEL")
    else:
        raise ValueError("mode must be 'sdk' or 'proxy'")

def mk_proxy_client() -> AsyncOpenAI:
    """
    Cliente para o LiteLLM Proxy (OpenAI-compatible).
    Necessário:
      LITELLM_PROXY_URL  (ex.: http://localhost:4000/v1)
      LITELLM_PROXY_KEY
    """
    base_url = os.getenv("LITELLM_PROXY_URL")
    api_key = os.getenv("LITELLM_PROXY_KEY")
    if not base_url or not api_key:
        raise RuntimeError("Missing LITELLM_PROXY_URL or LITELLM_PROXY_KEY for mode=proxy")
    timeout = float(os.getenv("REQUEST_TIMEOUT", "60"))
    return AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

def load_scenario(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# -------------------------
# Chamadas: LiteLLM SDK
# -------------------------
async def litellm_call(
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
    tools: Optional[List[Dict[str, Any]]] = None,
    stream: bool = False,
) -> Dict[str, Any]:
    """
    Faz uma chamada via LiteLLM SDK (direto no provedor).
    - Em stream: retorna texto concatenado e TTFT (se possível).
    - Em não-stream: retorna usage (quando o provedor suporta).
    Observação: as chaves do provedor vêm do ambiente (ex.: OPENAI_API_KEY, ANTHROPIC_API_KEY etc).
    """
    t0 = time.perf_counter()
    ttft_s: Optional[float] = None
    text_out = ""
    usage_in = 0
    usage_out = 0

    try:
        if stream:
            # streaming async
            first = True
            resp_stream = await litellm.acompletion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                stream=True,
                request_timeout=float(os.getenv("REQUEST_TIMEOUT", "60")),
            )
            async for chunk in resp_stream:
                # Primeiro delta com conteúdo -> TTFT
                content = None
                try:
                    content = chunk["choices"][0]["delta"].get("content")
                except Exception:
                    pass
                if content:
                    if first:
                        ttft_s = time.perf_counter() - t0
                        first = False
                    text_out += content

            # Em streams, a maioria dos provedores não traz usage:
            usage_in = usage_out = 0

        else:
            # não-stream
            resp = await litellm.acompletion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                request_timeout=float(os.getenv("REQUEST_TIMEOUT", "60")),
            )
            # texto
            text_out = (resp["choices"][0]["message"].get("content") or "").strip()
            # usage (quando normalizado pelo litellm/provedor)
            u = resp.get("usage") or {}
            usage_in = int(u.get("prompt_tokens") or u.get("input_tokens") or 0)
            usage_out = int(u.get("completion_tokens") or u.get("output_tokens") or 0)

        t1 = time.perf_counter()
        return {
            "status": "ok",
            "latency_ms": round((t1 - t0) * 1000, 2),
            "ttft_ms": round(ttft_s * 1000, 2) if ttft_s is not None else None,
            "input_tokens": usage_in,
            "output_tokens": usage_out,
            "text_len": len(text_out),
            "error": ""
        }

    except Exception as e:
        t1 = time.perf_counter()
        return {
            "status": "error",
            "latency_ms": round((t1 - t0) * 1000, 2),
            "ttft_ms": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "text_len": 0,
            "error": str(e)[:500],
        }

# -------------------------
# Chamadas: LiteLLM Proxy
# -------------------------
async def proxy_call(
    proxy_client: AsyncOpenAI,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
    tools: Optional[List[Dict[str, Any]]] = None,
    stream: bool = False,
) -> Dict[str, Any]:
    """
    Faz uma chamada ao LiteLLM Proxy (OpenAI-compatible) usando o cliente AsyncOpenAI.
    Em stream: mede TTFT.
    Em não-stream: tenta ler usage.
    """
    t0 = time.perf_counter()
    ttft_s: Optional[float] = None
    text_out = ""
    usage_in = 0
    usage_out = 0

    try:
        if stream:
            first_token_time = None
            content_chunks: List[str] = []
            async with await proxy_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                stream=True,
            ) as stream_resp:
                async for event in stream_resp:
                    if event.choices and event.choices[0].delta and event.choices[0].delta.content:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        content_chunks.append(event.choices[0].delta.content)

            if first_token_time:
                ttft_s = first_token_time - t0

            # Muitos proxies não retornam usage em stream:
            usage_in = usage_out = 0
            text_out = "".join(content_chunks)

        else:
            resp = await proxy_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
            )
            text_out = (resp.choices[0].message.content or "").strip()

            usage = getattr(resp, "usage", None)
            usage_in = int(getattr(usage, "prompt_tokens", None) or 0)
            usage_out = int(getattr(usage, "completion_tokens", None) or 0)

        t1 = time.perf_counter()
        return {
            "status": "ok",
            "latency_ms": round((t1 - t0) * 1000, 2),
            "ttft_ms": round(ttft_s * 1000, 2) if ttft_s is not None else None,
            "input_tokens": usage_in,
            "output_tokens": usage_out,
            "text_len": len(text_out),
            "error": ""
        }

    except Exception as e:
        t1 = time.perf_counter()
        return {
            "status": "error",
            "latency_ms": round((t1 - t0) * 1000, 2),
            "ttft_ms": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "text_len": 0,
            "error": str(e)[:500],
        }

# -------------------------
# Execução de uma vez + worker
# -------------------------
async def run_once(mode: str, proxy_client: Optional[AsyncOpenAI], model: str, scenario: Dict[str, Any],
                   temperature: float, max_tokens: int, stream: bool=False) -> Dict[str, Any]:
    """
    Despacha para o caminho correto (SDK vs Proxy).
    """
    messages = scenario["messages"]
    tools = scenario.get("tools")
    if mode == "sdk":
        return await litellm_call(model, messages, temperature, max_tokens, tools, stream)
    elif mode == "proxy":
        assert proxy_client is not None
        return await proxy_call(proxy_client, model, messages, temperature, max_tokens, tools, stream)
    else:
        raise ValueError("mode must be 'sdk' or 'proxy'")

async def worker(n: int, sem: asyncio.Semaphore, mode: str, proxy_client: Optional[AsyncOpenAI], model: str,
                 scenario: Dict[str, Any], temperature: float, max_tokens: int, stream: bool,
                 results: List[Dict[str, Any]], writer: csv.DictWriter, wlock: asyncio.Lock, csv_file):
    """
    Executa uma requisição e:
      - adiciona no array `results`
      - escreve uma linha no CSV imediatamente (flush para leitura ao vivo)
    """
    async with sem:
        res = await run_once(mode, proxy_client, model, scenario, temperature, max_tokens, stream)
        results.append(res)
        # escreve CSV ao vivo
        async with wlock:
            writer.writerow(res)
            csv_file.flush()

# -------------------------
# Agregação
# -------------------------
def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    latencies = [r["latency_ms"] for r in rows if r["status"]=="ok"]
    ttfts = [r["ttft_ms"] for r in rows if r["status"]=="ok" and r["ttft_ms"] is not None]
    errs = [r for r in rows if r["status"]=="error"]

    def pct(values, p):
        if not values:
            return None
        values_sorted = sorted(values)
        k = (len(values_sorted)-1) * (p/100.0)
        f = math.floor(k); c = math.ceil(k)
        if f == c:
            return values_sorted[int(k)]
        d0 = values_sorted[f] * (c - k)
        d1 = values_sorted[c] * (k - f)
        return d0 + d1

    return {
        "ok": len(rows)-len(errs),
        "errors": len(errs),
        "p50_ms": round(pct(latencies, 50), 2) if latencies else None,
        "p95_ms": round(pct(latencies, 95), 2) if latencies else None,
        "ttft_p50_ms": round(pct(ttfts, 50), 2) if ttfts else None,
        "ttft_p95_ms": round(pct(ttfts, 95), 2) if ttfts else None,
    }

# -------------------------
# Main CLI
# -------------------------
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sdk","proxy"], required=True,
                        help="sdk = LiteLLM SDK direto no provedor; proxy = LiteLLM Proxy (OpenAI-compatible)")
    parser.add_argument("--scenario", choices=["chat","stream","tools"], required=True)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=float(os.getenv("TEMPERATURE", "0.2")))
    parser.add_argument("--max_tokens", type=int, default=int(os.getenv("MAX_TOKENS", "128")))
    parser.add_argument("--run_id", type=str, default=None,
                        help="Identificador opcional do run (para nomear arquivos e dashboards)")
    args = parser.parse_args()

    scenario_path = f"scenarios/{args.scenario}.json"
    scenario = load_scenario(scenario_path)
    stream = (args.scenario == "stream")

    model = get_model(args.mode)
    if not model:
        raise RuntimeError(f"Missing model name. Set {'LITELLM_SDK_MODEL' if args.mode=='sdk' else 'PROXY_MODEL'}.")

    proxy_client = mk_proxy_client() if args.mode == "proxy" else None

    # timestamp/id do run (p/ nomes de arquivos)
    ts = args.run_id or time.strftime("%Y%m%d-%H%M%S")

    # prepara CSV para escrita ao vivo
    csv_path = REPORTS_DIR / f"results_{args.mode}_{args.scenario}_{ts}.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    # usamos o superset de campos retornados por run_once
    fieldnames = ["status","latency_ms","ttft_ms","input_tokens","output_tokens","text_len","error"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    csv_file.flush()

    sem = asyncio.Semaphore(args.concurrency)
    results: List[Dict[str, Any]] = []
    wlock = asyncio.Lock()  # lock para escrita no CSV

    tasks = [
        asyncio.create_task(
            worker(i, sem, args.mode, proxy_client, model, scenario,
                   args.temperature, args.max_tokens, stream, results, writer, wlock, csv_file)
        )
        for i in range(args.requests)
    ]

    t0 = time.perf_counter()
    await asyncio.gather(*tasks)
    t1 = time.perf_counter()

    # fecha CSV
    try:
        csv_file.close()
    except Exception:
        pass

    # preços e custo
    pricing = load_pricing()
    total_in = sum(r["input_tokens"] for r in results)
    total_out = sum(r["output_tokens"] for r in results)
    cost = estimate_cost(model, total_in, total_out, pricing)

    # summary
    agg = summarize_rows(results)
    runtime = (t1 - t0) or 1e-9
    throughput = round(args.requests / runtime, 2)
    summary_path = REPORTS_DIR / f"summary_{args.mode}_{args.scenario}_{ts}.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"# Summary — {args.mode} / {args.scenario}\n\n")
        f.write(f"- Requests: {args.requests}\n")
        f.write(f"- Concurrency: {args.concurrency}\n")
        f.write(f"- Model: {model}\n")
        f.write(f"- Runtime: {round(runtime,2)}s\n")
        f.write(f"- Throughput: {throughput} req/s\n")
        f.write(f"- OK: {agg['ok']}  Errors: {agg['errors']}\n")
        f.write(f"- p50: {agg['p50_ms']} ms  p95: {agg['p95_ms']} ms\n")
        if agg["ttft_p50_ms"] is not None:
            f.write(f"- TTFT p50: {agg['ttft_p50_ms']} ms  p95: {agg['ttft_p95_ms']} ms\n")
        f.write(f"- Tokens in/out: {total_in}/{total_out}\n")
        f.write(f"- Cost est.: ${cost:.4f}\n")

    print(f"Wrote {csv_path} and {summary_path}")

    if proxy_client:
        await proxy_client.close()

if __name__ == "__main__":
    asyncio.run(main())
