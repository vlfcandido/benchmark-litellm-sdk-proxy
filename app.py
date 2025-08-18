#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# app.py — LiteLLM: SDK vs Proxy com placares (%), ECDF, progresso/ETA e resumos curtos por GPT em layout legível.
# Execução: streamlit run app.py
#
# Requisitos:
#   pip install streamlit pandas plotly python-dotenv openai
#
# Observações:
# - O runner CLI (run_bench.py) executa os cenários.
# - Mostramos KPIs e gráficos; GPT gera resumos curtos, com scoreboard em TABELA e 3 blocos (TL;DR, Impacto, Limitações).
# - Os prompts citam explicitamente "LiteLLM SDK" e "LiteLLM Proxy".
# - Ajuste pedido: remover emojis dos prompts e reforçar interpretação de Consistência (gap menor = mais estável)
#   e prioridade de Confiabilidade na decisão final.

import os
import re
import time
import glob
import subprocess
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

# =============================================================================
# Configuração base
# =============================================================================

load_dotenv()

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

BENCH = "run_bench.py"
st.set_page_config(page_title="LiteLLM — SDK vs Proxy", layout="wide")
st.title("LiteLLM — SDK vs Proxy")

# Estado para ETA histórico (para estimar duração das próximas execuções)
if "eta_hist" not in st.session_state:
    st.session_state.eta_hist: Dict[Tuple[str, str, str, str], List[float]] = {}

# =============================================================================
# Utilitários básicos
# =============================================================================

def ensure_default_scenarios() -> None:
    """
    Garante que existam cenários mínimos no diretório ./scenarios, caso não estejam presentes.
    Isso facilita o 'primeiro run' sem precisar criar arquivos manualmente.
    """
    scenarios_dir = Path("scenarios")
    scenarios_dir.mkdir(exist_ok=True)
    defaults = {
        "chat.json": {
            "messages": [
                {"role": "user", "content": "Você é analista de suporte. Explique em 3 frases como reiniciar com segurança o serviço de faturamento sem indisponibilizar a API."}
            ]
        },
        "stream.json": {
            "messages": [
                {"role": "user", "content": "Gere um plano em 6-8 passos para migrar uma API REST para GraphQL, destacando prós e contras."}
            ]
        },
        "tools.json": {
            "messages": [
                {"role": "user", "content": "Dado um CEP do Brasil, busque cidade e UF e retorne como JSON (simule function calling)."}
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_cep",
                        "description": "Consulta de CEP para obter cidade e UF",
                        "parameters": {
                            "type": "object",
                            "properties": {"cep": {"type": "string", "description": "CEP com 8 dígitos"}},
                            "required": ["cep"]
                        }
                    }
                }
            ]
        }
    }
    for fname, payload in defaults.items():
        path = scenarios_dir / fname
        if not path.exists():
            with open(path, "w", encoding="utf-8") as f:
                import json
                json.dump(payload, f, indent=2, ensure_ascii=False)

def parse_wrote_paths(stdout: str) -> Tuple[str, str]:
    """
    Extrai do stdout a linha "Wrote <csv> and <summary>" produzida pelo runner, se houver.
    Retorna (csv_path, summary_path) ou ('','') se não encontrar.
    """
    m = re.search(r"Wrote\s+(.+?)\s+and\s+(.+?)\s*$", stdout.strip())
    if m:
        return m.group(1), m.group(2)
    return "", ""

def last_csv_for(mode: str, scenario: str) -> str:
    """
    Se o runner não imprimir o caminho do CSV, tenta recuperar o último arquivo pelos padrões de nome.
    """
    files = sorted(glob.glob(f"reports/results_{mode}_{scenario}_*.csv"))
    return files[-1] if files else ""

def coerce_df_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte colunas numéricas e garante que 'status' seja string.
    Evita erros de tipo quando o CSV tem valores vazios/mistos.
    """
    for col in ["latency_ms", "ttft_ms", "input_tokens", "output_tokens"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "status" in df.columns:
        df["status"] = df["status"].astype(str)
    return df

def resumo(df: pd.DataFrame) -> Dict:
    """
    Gera um resumo com contagem de OK/erros, p50/p95 de latência (em OK),
    TTFT p50 (se existir), e soma de tokens de entrada/saída.
    """
    df = coerce_df_types(df)
    if df.empty:
        return dict(ok=0, errors=0, p50=None, p95=None, ttft=None, in_tok=0, out_tok=0)
    ok = int((df.status == "ok").sum())
    errors = int((df.status != "ok").sum())
    lat_ok = df.loc[df.status == "ok", "latency_ms"].dropna()
    p50 = float(lat_ok.quantile(0.5)) if not lat_ok.empty else None
    p95 = float(lat_ok.quantile(0.95)) if not lat_ok.empty else None
    ttft = None
    if "ttft_ms" in df.columns:
        ttfts = df.loc[(df.status == "ok") & (df["ttft_ms"].notna()) & (df["ttft_ms"] > 0), "ttft_ms"]
        ttft = float(ttfts.quantile(0.5)) if not ttfts.empty else None
    in_tok = int(pd.to_numeric(df.get("input_tokens", pd.Series([0])), errors="coerce").fillna(0).sum())
    out_tok = int(pd.to_numeric(df.get("output_tokens", pd.Series([0])), errors="coerce").fillna(0).sum())
    return dict(ok=ok, errors=errors, p50=p50, p95=p95, ttft=ttft, in_tok=in_tok, out_tok=out_tok)

# ---------- Indicadores derivados ----------
def _pct_diff_better_lower(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """
    Diferença percentual quando 'quanto menor, melhor'.
    Retorna ((b - a) / b) * 100. Ex.: a=10, b=20 -> 50% melhor.
    """
    if a is None or b in (None, 0): return None
    return (b - a) / b * 100.0

def _pct_diff_better_higher(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """
    Diferença percentual quando 'quanto maior, melhor'.
    Retorna ((a - b) / b) * 100.
    """
    if a is None or b in (None, 0): return None
    return (a - b) / b * 100.0

def _safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b in (None, 0): return None
    return a / b

def derive_indicators(r_sdk: Dict, r_proxy: Dict, thr_sdk: float, thr_proxy: float) -> Dict:
    """
    Calcula taxa de sucesso, gap de consistência (p95/p50) e deltas percentuais básicos.
    OBS: gap menor = mais consistente.
    """
    total_sdk = r_sdk["ok"] + r_sdk["errors"]
    total_proxy = r_proxy["ok"] + r_proxy["errors"]
    succ_sdk = (r_sdk["ok"] / total_sdk * 100.0) if total_sdk > 0 else None
    succ_proxy = (r_proxy["ok"] / total_proxy * 100.0) if total_proxy > 0 else None
    gap_sdk = _safe_div(r_sdk["p95"], r_sdk["p50"])
    gap_proxy = _safe_div(r_proxy["p95"], r_proxy["p50"])
    pct_p50_sdk_vs_proxy = _pct_diff_better_lower(r_sdk["p50"], r_proxy["p50"])
    pct_p95_sdk_vs_proxy = _pct_diff_better_lower(r_sdk["p95"], r_proxy["p95"])
    pct_thr_sdk_vs_proxy = _pct_diff_better_higher(thr_sdk, thr_proxy)
    pct_succ_sdk_vs_proxy = None
    if succ_sdk is not None and succ_proxy not in (None, 0):
        pct_succ_sdk_vs_proxy = _pct_diff_better_higher(succ_sdk, succ_proxy)
    pct_gap_sdk_vs_proxy = None
    if gap_sdk is not None and gap_proxy not in (None, 0):
        pct_gap_sdk_vs_proxy = _pct_diff_better_lower(gap_sdk, gap_proxy)
    return dict(
        success_rate=dict(sdk=succ_sdk, proxy=succ_proxy, delta_pct_vs_proxy=pct_succ_sdk_vs_proxy),
        consistency_gap=dict(sdk=gap_sdk, proxy=gap_proxy, delta_pct_vs_proxy=pct_gap_sdk_vs_proxy),
        deltas_pct=dict(p50=pct_p50_sdk_vs_proxy, p95=pct_p95_sdk_vs_proxy, throughput=pct_thr_sdk_vs_proxy),
    )

# ---------- Histórico ETA ----------
def _hist_key(scenario: str, perfil: str, model_key: str, mode: str) -> Tuple[str, str, str, str]:
    return (scenario, perfil, model_key, mode)

def _hist_avg_seconds(key: Tuple[str, str, str, str]) -> Optional[float]:
    vals = st.session_state.eta_hist.get(key) or []
    if not vals: return None
    s = sorted(vals); n = max(1, int(len(s)*0.8))
    return sum(s[:n]) / n

def _hist_add_seconds(key: Tuple[str, str, str, str], elapsed: float) -> None:
    st.session_state.eta_hist.setdefault(key, []).append(float(elapsed))
    if len(st.session_state.eta_hist[key]) > 20:
        st.session_state.eta_hist[key] = st.session_state.eta_hist[key][-20:]

# ---------- Execução com progresso ----------
def run_once_with_progress(mode: str, scenario: str, requests: int, concurrency: int,
                           temperature: float, max_tokens: int, model_key: str,
                           progress_area, expected_seconds: Optional[float] = None, overall=None
                           ) -> Tuple[str, float, str]:
    """
    Executa um cenário via runner CLI e mostra progresso/ETA aproximado.
    Retorna (csv_path, elapsed_seconds, stdout_completo).
    """
    cmd = [
        "python", BENCH, "--mode", mode, "--scenario", scenario,
        "--requests", str(requests), "--concurrency", str(concurrency),
        "--temperature", str(temperature), "--max_tokens", str(max_tokens),
    ]
    env = os.environ.copy()
    if mode == "sdk":
        env["LITELLM_SDK_MODEL"] = model_key
    else:
        env["PROXY_MODEL"] = model_key
        url = env.get("LITELLM_PROXY_URL")
        if url and not url.rstrip("/").endswith("/v1"):
            env["LITELLM_PROXY_URL"] = url.rstrip("/") + "/v1"

    title_ph = progress_area.empty()
    bar = progress_area.progress(0)
    eta_ph = progress_area.empty()
    title_ph.markdown(f"**{scenario} — {('LiteLLM SDK' if mode=='sdk' else 'LiteLLM Proxy')}**")

    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    output_lines: List[str] = []
    t0 = time.perf_counter()

    if overall:
        if "bar" not in overall: overall["bar"] = overall["container"].progress(0)
        if "eta" not in overall: overall["eta"] = overall["container"].empty()

    last_ui = 0.0
    while True:
        line = proc.stdout.readline()
        if line: output_lines.append(line)
        if proc.poll() is not None:
            rest = proc.stdout.read()
            if rest: output_lines.append(rest)
            break

        elapsed = time.perf_counter() - t0
        if expected_seconds and expected_seconds > 0:
            pct = min(int((elapsed / expected_seconds) * 100), 95)
            remaining = max(expected_seconds - elapsed, 0.0)
            eta_ph.markdown(f"{elapsed:0.1f}s decorridos • ETA ~ {remaining:0.1f}s")
        else:
            pct = int((elapsed * 13) % 95) + 5
            eta_ph.markdown(f"{elapsed:0.1f}s decorridos (estimando…)")

        bar.progress(max(1, min(pct, 99)))
        if overall:
            steps_done = overall["done"]; total_steps = overall["total"]
            frac_current = (pct / 100.0)
            overall_pct = int(((steps_done + frac_current) / total_steps) * 100)
            overall["bar"].progress(min(overall_pct, 99))
            if expected_seconds and overall.get("rem_secs_total") is not None:
                rem_current = max(expected_seconds - elapsed, 0.0)
                remaining_total = rem_current + overall["rem_secs_total"]
                overall["eta"].markdown(f"Progresso geral: {overall_pct}% • ETA ~ {remaining_total:0.1f}s")

        now = time.perf_counter()
        if now - last_ui < 0.15: time.sleep(0.05)
        last_ui = now

    elapsed = time.perf_counter() - t0
    bar.progress(100)
    eta_ph.markdown(f"Concluído em {elapsed:0.1f}s")

    full_out = "".join(output_lines)
    csv_path, _summary = parse_wrote_paths(full_out)
    if not csv_path: csv_path = last_csv_for(mode, scenario)
    if proc.returncode not in (0, None): raise RuntimeError(f"{mode}/{scenario} falhou:\n{full_out}")
    return csv_path, elapsed, full_out

# ---------- UI KPIs ----------
def render_kpi_grid(title: str, items: List[Tuple[str, Optional[float], str, str]]) -> None:
    """
    Renderiza uma grade simples de KPIs com título, tooltip e valor.
    """
    boxes = []
    for label, val, unit, tip in items:
        val_txt = "--" if val is None else (f"{val:.1f}" if unit == "ms" else f"{val:.2f}")
        boxes.append(f"""
          <div class="stat-col">
            <div class="stat-title"><span title="{tip}">{label}</span></div>
            <div class="stat-value">{val_txt}</div>
          </div>
        """)
    html = f"""
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <style>
      :root {{ --fg:#111; --muted:#666; --border:#e5e5e5; --bg:#fff; }}
      body {{ margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background:transparent; }}
      .title {{ font-weight:600; margin: 0 0 8px 0; }}
      .grid {{ display:grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
      .stat-col {{ border:1px solid var(--border); border-radius:8px; padding:12px; background:var(--bg); }}
      .stat-title {{ font-size:0.9rem; color:var(--muted); margin-bottom:6px; }}
      .stat-title span {{ border-bottom:1px dotted #999; cursor: help; }}
      .stat-value {{ font-size:1.25rem; font-weight:600; color:var(--fg); }}
    </style>
  </head>
  <body>
    <div class="title">{title}</div>
    <div class="grid">
      {''.join(boxes)}
    </div>
  </body>
</html>
"""
    height = 160 if len(items) > 3 else 120
    components.html(html, height=height, scrolling=False)

# =============================================================================
# Resumos via GPT — PROMPTS AJUSTADOS (sem emojis, com ênfase em consistência e confiabilidade)
# =============================================================================

def render_practical_gpt_summary(scenario_name: str, scenario_desc: str, data: Dict, meta: Dict) -> None:
    """
    Gera um resumo detalhado do cenário (LiteLLM SDK vs LiteLLM Proxy) a partir dos dados do run.
    Prompt ajustado: sem emojis, explicitando que Consistência = p95/p50 (menor é melhor)
    e que taxa de sucesso tem peso decisivo em produção.
    """
    import json as _json
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model_for_summary = os.getenv("SUMMARY_MODEL", "gpt-4o-mini")
    if not api_key:
        st.warning("Para gerar o resumo automático, defina OPENAI_API_KEY (e opcionalmente OPENAI_BASE_URL/SUMMARY_MODEL).")
        return

    prompt = f"""Você é um arquiteto de sistemas especialista em análise de performance de APIs e inferência de LLM. Analise criticamente os dados do cenário "{scenario_name}" para fornecer uma recomendação técnica clara sobre LiteLLM SDK vs LiteLLM Proxy.

REGRAS DE AVALIAÇÃO IMPORTANTES:
- Consistência é medida por GAP = p95/p50; quanto MENOR o GAP, mais estável.
- Priorize Confiabilidade (Taxa de Sucesso). Em produção, baixa confiabilidade invalida ganhos marginais de latência/throughput.
- Interprete Throughput no contexto do tempo total do cenário.
- Se houver TTFT no cenário "stream", inclua na análise.
- Evite conclusões genéricas; aponte trade-offs e implicações práticas.

CONTEXTO DO CENÁRIO:
- Nome: "{scenario_name}" — {scenario_desc}
- Ambiente/Perfil: {meta.get('perfil', 'não especificado')}
- Modelo: {meta.get('model_key', 'não especificado')}
- Concorrência: {data.get('settings', {}).get('concurrency', 'N/A')} paralelas
- Total de requisições: {data.get('settings', {}).get('requests', 'N/A')}

DADOS COLETADOS (JSON):
{_json.dumps(data, ensure_ascii=False, indent=2)}

RETORNE APENAS o Markdown no formato abaixo (substitua todos os placeholders):

### Scoreboard — {scenario_name}

| Métrica | LiteLLM SDK | LiteLLM Proxy | Diferença (%) | Vencedor | Significância |
|---------|-------------|---------------|---------------|----------|---------------|
| Latência p50 | {{sdk_p50}}ms | {{proxy_p50}}ms | {{delta_p50}}% | {{winner_p50}} | {{sig_p50}} |
| Latência p95 | {{sdk_p95}}ms | {{proxy_p95}}ms | {{delta_p95}}% | {{winner_p95}} | {{sig_p95}} |
| Throughput | {{sdk_thr}} req/s | {{proxy_thr}} req/s | {{delta_thr}}% | {{winner_thr}} | {{sig_thr}} |
| Consistência (p95/p50) | {{sdk_gap}}x | {{proxy_gap}}x | {{delta_gap}}% | {{winner_gap}} | {{sig_gap}} |
| Taxa de sucesso | {{sdk_success}}% | {{proxy_success}}% | {{delta_success}}pp | {{winner_success}} | {{sig_success}} |
| Tempo de execução | {{sdk_time}}s | {{proxy_time}}s | {{delta_time}}% | {{winner_time}} | {{sig_time}} |

Legenda de Significância: Crítica (>20%) | Moderada (5–20%) | Baixa (<5%)

### Análise Técnica
Vencedor do cenário: {{vencedor_geral}} — {{vantagem_principal}}

Pontos críticos:
- {{analise_performance_1}}
- {{analise_performance_2}}
- {{analise_performance_3}}

Implicações arquiteturais:
- {{implicacao_arquitetural_1}}
- {{implicacao_arquitetural_2}}

### Impacto no Usuário Final
Para usuários finais:
{{impacto_usuario_final}}

Para desenvolvedores/operação:
{{impacto_dev_ops}}

Cenários de uso recomendados:
- {{vencedor_geral}} -> {{casos_uso_recomendados}}
- {{perdedor_geral}} -> {{casos_uso_alternativos}}

### Considerações de Implementação
Fatores de decisão críticos:
{{fatores_decisao}}

Riscos identificados:
{{riscos_identificados}}

Recomendações de otimização:
{{recomendacoes_otimizacao}}

### Limitações da Análise
{{limitacoes_analise}}

Decisão recomendada:
{{decisao_final_detalhada}}

Próximos passos sugeridos:
{{proximos_passos}}

INSTRUÇÕES CRÍTICAS:
1) Substitua TODOS os placeholders {{...}} com valores reais dos dados.
2) Para "Consistência", lembre que gap menor é melhor.
3) Classifique "Significância" como: Crítica (>20%), Moderada (5–20%), Baixa (<5%).
4) Se a Taxa de Sucesso de um lado for < 95%, NÃO declare esse lado como vencedor geral para produção; explique.
5) Seja específico com números; não responda com texto genérico.
"""

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=model_for_summary,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1200,
        )
        text = (resp.choices[0].message.content or "").strip()
        st.markdown(text)
    except Exception as e:
        st.error("Falha ao gerar o resumo do cenário com GPT.")
        st.exception(e)

def render_technical_gpt_overall_summary(results: List[Dict], meta: Dict) -> None:
    """
    Gera um resumo executivo consolidado (todos os cenários).
    Prompt ajustado: sem emojis e com orientação explícita de priorizar confiabilidade.
    """
    import json as _json
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model_for_summary = os.getenv("SUMMARY_MODEL", "gpt-4o-mini")
    if not api_key:
        st.warning("Para gerar o resumo técnico final, defina OPENAI_API_KEY (e opcionalmente OPENAI_BASE_URL/SUMMARY_MODEL).")
        return

    prompt = f"""Você é um CTO/Arquiteto Senior analisando benchmarks para decidir entre LiteLLM SDK vs LiteLLM Proxy em produção. Forneça recomendação executiva baseada em TODOS os cenários.

PRINCÍPIOS:
- Confiabilidade (Taxa de sucesso) tem precedência em produção.
- Consistência = p95/p50; menor é melhor.
- Considere médias entre cenários e destaque outliers.

CONTEXTO:
- Perfil de teste: {meta.get('perfil', 'não especificado')}
- Modelo testado: {meta.get('model_key', 'não especificado')}
- Configurações: {meta.get('settings', {})}
- Timestamp: {meta.get('timestamp', 'não especificado')}
- Total de cenários: {len(results)}

DADOS (JSON):
{_json.dumps(results, ensure_ascii=False, indent=2)}

RETORNE APENAS o Markdown no formato abaixo:

# Recomendação Estratégica: LiteLLM SDK vs Proxy

## Panorama Executivo
Perfil: {meta.get('perfil', 'N/A')} | Modelo: {meta.get('model_key', 'N/A')} | Concorrência: {meta.get('settings', {}).get('concurrency', 'N/A')} | Total de Requisições: {meta.get('settings', {}).get('requests', 'N/A')}

## Scoreboard Consolidado (média dos cenários)
| Métrica | LiteLLM SDK | LiteLLM Proxy | Diferença | Vencedor Geral |
|---------|-------------|---------------|-----------|----------------|
| Latência Mediana | {{sdk_p50_avg}}ms | {{proxy_p50_avg}}ms | {{delta_p50_avg}}% | {{winner_p50_avg}} |
| Latência p95 | {{sdk_p95_avg}}ms | {{proxy_p95_avg}}ms | {{delta_p95_avg}}% | {{winner_p95_avg}} |
| Throughput | {{sdk_thr_avg}} req/s | {{proxy_thr_avg}} req/s | {{delta_thr_avg}}% | {{winner_thr_avg}} |
| Consistência (p95/p50) | {{sdk_gap_avg}}x | {{proxy_gap_avg}}x | {{delta_gap_avg}}% | {{winner_gap_avg}} |
| Confiabilidade | {{sdk_success_avg}}% | {{proxy_success_avg}}% | {{delta_success_avg}}pp | {{winner_success_avg}} |

Matriz de cenários:
{{matriz_cenarios_desempenho}}

## Recomendação Principal
Vencedor Geral: {{vencedor_arquitetural}}

Justificativa técnica:
{{justificativa_tecnica_detalhada}}

Vantagens do vencedor:
{{vantagens_competitivas}}

Trade-offs:
{{tradeoffs_identificados}}

## Impactos por Contexto de Uso
Chat/Assistentes:
{{recomendacao_chatbot}}

Geração longa (stream):
{{recomendacao_stream}}

Function calling (tools):
{{recomendacao_tools}}

## Considerações de Produção
Arquiteturais:
{{consideracoes_arquiteturais}}

Operacionais:
{{consideracoes_operacionais}}

Custo vs Performance:
{{analise_custo_performance}}

Escalabilidade:
{{analise_escalabilidade}}

## Fatores de Risco
Riscos do {{vencedor_arquitetural}}:
{{riscos_vencedor}}

Mitigações:
{{mitigacoes_recomendadas}}

## Plano de Implementação
Fase 1 - Piloto:
{{plano_fase1}}

Fase 2 - Produção:
{{plano_fase2}}

Fase 3 - Otimização contínua:
{{plano_fase3}}

## Decisão Executiva Final
{{decisao_executiva_final}}

Condições de Revisão:
{{condicoes_revisao}}

INSTRUÇÕES:
1) Substitua todos os placeholders {{...}} com cálculos reais.
2) Se a confiabilidade média de um lado < 95%, não recomende esse lado para produção sem mitigação explícita.
3) Destaque outliers e explique se distorcem as médias.
"""

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=model_for_summary,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000,
        )
        text = (resp.choices[0].message.content or "").strip()
        st.markdown(text)
    except Exception as e:
        st.error("Falha ao gerar o resumo técnico final com GPT.")
        st.exception(e)

def winner(a: Optional[float], b: Optional[float], lower_is_better: bool = True) -> str:
    """
    Escolhe vencedor para um par de métricas.
    lower_is_better=True para latências/tempos; False para throughput/sucesso.
    """
    if a is None and b is None: return "-"
    if a is None: return "LiteLLM Proxy"
    if b is None: return "LiteLLM SDK"
    if lower_is_better: return "LiteLLM SDK" if a < b else "LiteLLM Proxy"
    else: return "LiteLLM SDK" if a > b else "LiteLLM Proxy"

# =============================================================================
# Sidebar — Configuração
# =============================================================================

ensure_default_scenarios()

with st.sidebar:
    st.header("Configuração")
    scenarios: List[str] = st.multiselect(
        "Cenários", ["chat", "stream", "tools"],
        default=["chat", "stream", "tools"], help="Selecione um ou mais cenários para rodar."
    )
    PERFIS = {
        "Dev rápido": {"requests": 20, "concurrency": 5, "temperature": 0.3, "max_tokens": 128, "desc": "Execução curta para validar rapidamente."},
        "PRD": {"requests": 100, "concurrency": 20, "temperature": 0.2, "max_tokens": 256, "desc": "Perfil estável, próximo do uso de produção."},
        "Estresse": {"requests": 500, "concurrency": 100, "temperature": 0.2, "max_tokens": 512, "desc": "Pressão alta para expor p95 e gargalos."},
    }
    perfil_nome = st.selectbox("Perfil", list(PERFIS.keys()), index=1,
                               help="Concurrency = requisições em paralelo. Temperature = aleatoriedade do texto.")
    perfil = PERFIS[perfil_nome]; st.caption(perfil["desc"])

    MODELS = [
        {"label": "OpenAI: gpt-4o-mini (recomendado)", "key": "openai/gpt-4o-mini", "blocked": False},
        {"label": "OpenAI: gpt-4o (em breve)", "key": "openai/gpt-4o", "blocked": True},
        {"label": "Google: Gemini 1.5 Flash (em breve)", "key": "google/gemini-1.5-flash", "blocked": True},
        {"label": "Google: Gemini 1.5 Pro (em breve)", "key": "google/gemini-1.5-pro", "blocked": True},
    ]
    model_choice = st.selectbox("Modelo (SDK e Proxy)", [m["label"] for m in MODELS], index=0,
                                help="Marcados como 'em breve' aparecem, mas não executam.")
    _selected = next(m for m in MODELS if m["label"] == model_choice)
    model_key = _selected["key"]; model_blocked = _selected["blocked"]
    if model_blocked:
        st.warning("Este modelo ainda não está disponível. Use o gpt-4o-mini para rodar os testes.")

    show_progress = st.toggle("Mostrar barra de progresso com ETA (beta)", value=True,
                              help="Mostra progresso por execução (SDK/Proxy) e um progresso geral com ETA estimado.")

# =============================================================================
# Explicação técnica dos cenários (com exemplos JSON)
# =============================================================================
with st.expander("Cenários: o que cada um mede (com exemplos)"):
    st.markdown(
        "- **chat**: Q&A curto (FAQ / suporte). Mede latência estável em prompts simples.\n"
        "- **stream**: resposta longa com streaming; mede **TTFT** (primeiro token) além da latência total.\n"
        "- **tools**: simula function calling; mede overhead de decisão/estrutura."
    )
    st.markdown("**chat.json**")
    st.code('''{
  "messages": [
    {"role": "user", "content": "Você é analista de suporte. Explique em 3 frases como reiniciar com segurança o serviço de faturamento sem indisponibilizar a API."}
  ]
}''', language="json")
    st.markdown("**stream.json**")
    st.code('''{
  "messages": [
    {"role": "user", "content": "Gere um plano em 6-8 passos para migrar uma API REST para GraphQL, destacando prós e contras."}
  ]
}''', language="json")
    st.markdown("**tools.json**")
    st.code('''{
  "messages": [
    {"role": "user", "content": "Dado um CEP do Brasil, busque cidade e UF e retorne como JSON (simule function calling)."}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "lookup_cep",
        "description": "Consulta de CEP para obter cidade e UF",
        "parameters": {
          "type": "object",
          "properties": { "cep": {"type": "string", "description": "CEP com 8 dígitos"} },
          "required": ["cep"]
        }
      }
    }
  ]
}''', language="json")

# =============================================================================
# Execução
# =============================================================================
st.markdown("### Execução")
start = st.button("Executar comparação (SDK vs Proxy)", type="primary", disabled=model_blocked)

METRIC_TIPS = {
    "p50 (ms)": "Mediana: 50% das chamadas são mais rápidas que este valor.",
    "p95 (ms)": "Cauda: 95% das chamadas são mais rápidas; mostra o pior caso comum.",
    "Throughput (req/s)": "Respostas concluídas por segundo. Maior é melhor.",
    "TTFT p50 (ms)": "Time To First Token: tempo até o primeiro token no streaming (mediana).",
}

SCENARIO_DESC = {
    "chat": "Pergunta/Resposta curta; latência estável em prompts simples.",
    "stream": "Resposta longa com streaming; mede TTFT e latência total.",
    "tools": "Function calling; mede overhead de decisão/estrutura para ferramentas.",
}

if start:
    try:
        st.toast(f"Iniciando: {', '.join(scenarios)} — perfil {perfil_nome}")
    except Exception:
        st.success(f"Iniciando: {', '.join(scenarios)} — perfil {perfil_nome}")

    if not scenarios:
        st.error("Selecione pelo menos um cenário.")
    else:
        overall_results: List[Dict] = []
        run_meta = {
            "timestamp": datetime.now().isoformat(),
            "perfil": perfil_nome,
            "settings": dict(
                concurrency=perfil["concurrency"],
                requests=perfil["requests"],
                temperature=perfil["temperature"],
                max_tokens=perfil["max_tokens"]
            ),
            "model_key": model_key
        }

        overall_container = st.container() if show_progress else None
        overall = None
        if show_progress:
            total_steps = len(scenarios) * 2
            rem_secs_total = 0.0
            for scenario in scenarios:
                key_sdk = _hist_key(scenario, perfil_nome, model_key, "sdk")
                key_proxy = _hist_key(scenario, perfil_nome, model_key, "proxy")
                est_sdk = _hist_avg_seconds(key_sdk) or 0.0
                est_proxy = _hist_avg_seconds(key_proxy) or 0.0
                rem_secs_total += est_sdk + est_proxy
            overall = {"container": overall_container, "total": total_steps, "done": 0, "rem_secs_total": rem_secs_total}

        for scenario in scenarios:
            with st.expander(f"Cenário: {scenario}", expanded=True):
                try:
                    ctx_status = st.status(f"Executando LiteLLM SDK e LiteLLM Proxy para {scenario}...", expanded=False)
                except Exception:
                    ctx_status = None

                # --- Executa SDK ---
                place_sdk = st.container()
                key_sdk = _hist_key(scenario, perfil_nome, model_key, "sdk")
                exp_sdk = _hist_avg_seconds(key_sdk)
                try:
                    if ctx_status: ctx_status.update(label="Executando LiteLLM SDK…")
                    csv_sdk, time_sdk, _ = run_once_with_progress(
                        "sdk", scenario, perfil["requests"], perfil["concurrency"],
                        perfil["temperature"], perfil["max_tokens"], model_key,
                        progress_area=place_sdk, expected_seconds=exp_sdk,
                        overall=overall if show_progress else None
                    )
                    _hist_add_seconds(key_sdk, time_sdk)
                    if overall:
                        overall["done"] += 1
                        if overall["rem_secs_total"] is not None and exp_sdk:
                            overall["rem_secs_total"] = max(overall["rem_secs_total"] - exp_sdk, 0.0)
                except Exception as e:
                    if ctx_status: ctx_status.update(label="Falha na execução (LiteLLM SDK)", state="error")
                    st.error(str(e)); continue

                # --- Executa Proxy ---
                place_proxy = st.container()
                key_proxy = _hist_key(scenario, perfil_nome, model_key, "proxy")
                exp_proxy_hist = _hist_avg_seconds(key_proxy)
                exp_proxy = exp_proxy_hist or time_sdk  # fallback: proxy ~ tempo do sdk anterior
                try:
                    if ctx_status: ctx_status.update(label="Executando LiteLLM Proxy…")
                    csv_proxy, time_proxy, _ = run_once_with_progress(
                        "proxy", scenario, perfil["requests"], perfil["concurrency"],
                        perfil["temperature"], perfil["max_tokens"], model_key,
                        progress_area=place_proxy, expected_seconds=exp_proxy,
                        overall=overall if show_progress else None
                    )
                    _hist_add_seconds(key_proxy, time_proxy)
                    if overall:
                        overall["done"] += 1
                        if overall["rem_secs_total"] is not None and exp_proxy:
                            overall["rem_secs_total"] = max(overall["rem_secs_total"] - exp_proxy, 0.0)
                    if ctx_status: ctx_status.update(label="Concluído", state="complete")
                except Exception as e:
                    if ctx_status: ctx_status.update(label="Falha na execução (LiteLLM Proxy)", state="error")
                    st.error(str(e)); continue

                # --- KPIs e gráficos ---
                df_sdk = pd.read_csv(csv_sdk); df_proxy = pd.read_csv(csv_proxy)
                r_sdk = resumo(df_sdk); r_proxy = resumo(df_proxy)
                thr_sdk = round((r_sdk["ok"] / max(1e-6, time_sdk)), 2)
                thr_proxy = round((r_proxy["ok"] / max(1e-6, time_proxy)), 2)
                derived = derive_indicators(r_sdk, r_proxy, thr_sdk, thr_proxy)

                st.markdown("**Indicadores principais**")
                sdk_items = [("p50 (ms)", r_sdk["p50"], "ms", METRIC_TIPS["p50 (ms)"]),
                             ("p95 (ms)", r_sdk["p95"], "ms", METRIC_TIPS["p95 (ms)"]),
                             ("Throughput (req/s)", thr_sdk, "req/s", METRIC_TIPS["Throughput (req/s)"])]
                if scenario == "stream":
                    sdk_items.append(("TTFT p50 (ms)", r_sdk["ttft"], "ms", METRIC_TIPS["TTFT p50 (ms)"]))
                render_kpi_grid("LiteLLM SDK", sdk_items)

                proxy_items = [("p50 (ms)", r_proxy["p50"], "ms", METRIC_TIPS["p50 (ms)"]),
                               ("p95 (ms)", r_proxy["p95"], "ms", METRIC_TIPS["p95 (ms)"]),
                               ("Throughput (req/s)", thr_proxy, "req/s", METRIC_TIPS["Throughput (req/s)"])]
                if scenario == "stream":
                    proxy_items.append(("TTFT p50 (ms)", r_proxy["ttft"], "ms", METRIC_TIPS["TTFT p50 (ms)"]))
                render_kpi_grid("LiteLLM Proxy", proxy_items)

                st.markdown("**Latência — distribuição (boxplot)**")
                df_s_ok = coerce_df_types(df_sdk[df_sdk["status"] == "ok"].copy())
                df_p_ok = coerce_df_types(df_proxy[df_proxy["status"] == "ok"].copy())
                df_s_ok["mode"] = "LiteLLM SDK"; df_p_ok["mode"] = "LiteLLM Proxy"
                df_box = pd.concat([df_s_ok[["mode", "latency_ms"]], df_p_ok[["mode", "latency_ms"]]], ignore_index=True)
                if not df_box.empty:
                    fig_box = px.box(df_box, x="mode", y="latency_ms", points=False,
                                     labels={"latency_ms": "Latência (ms)", "mode": "Modo"})
                    fig_box.update_layout(showlegend=False)
                    st.plotly_chart(fig_box, use_container_width=True)
                else:
                    st.info("Sem dados OK para exibir boxplot.")

                st.markdown("**Latência — ECDF (curva acumulada)**")
                if not df_s_ok.empty or not df_p_ok.empty:
                    fig_ecdf = go.Figure()
                    if not df_s_ok.empty:
                        dd = df_s_ok["latency_ms"].sort_values().reset_index(drop=True)
                        y = (dd.rank(method="first") / len(dd)).values
                        fig_ecdf.add_trace(go.Scatter(x=dd, y=y, mode="lines", name="LiteLLM SDK"))
                    if not df_p_ok.empty:
                        dd = df_p_ok["latency_ms"].sort_values().reset_index(drop=True)
                        y = (dd.rank(method="first") / len(dd)).values
                        fig_ecdf.add_trace(go.Scatter(x=dd, y=y, mode="lines", name="LiteLLM Proxy"))
                    fig_ecdf.update_layout(xaxis_title="Latência (ms)", yaxis_title="Percentil (0–1)", legend_title_text="")
                    st.plotly_chart(fig_ecdf, use_container_width=True)
                else:
                    st.info("Sem dados OK para exibir ECDF.")

                st.markdown("**Throughput (req/s)**")
                thru_df = pd.DataFrame({"Modo": ["LiteLLM SDK", "LiteLLM Proxy"], "Throughput": [thr_sdk, thr_proxy]})
                fig_thr = go.Figure(data=[go.Bar(x=thru_df["Modo"], y=thru_df["Throughput"])])
                fig_thr.update_layout(yaxis_title="req/s", xaxis_title="")
                st.plotly_chart(fig_thr, use_container_width=True)

                # Vencedores pontuais das métricas principais
                win_p50 = "LiteLLM SDK" if (r_proxy["p50"] is None or (r_sdk["p50"] or float("inf")) < (r_proxy["p50"] or float("inf"))) else "LiteLLM Proxy"
                win_p95 = "LiteLLM SDK" if (r_proxy["p95"] is None or (r_sdk["p95"] or float("inf")) < (r_proxy["p95"] or float("inf"))) else "LiteLLM Proxy"
                win_thr = "LiteLLM SDK" if (thr_sdk >= thr_proxy) else "LiteLLM Proxy"

                # Payload consolidado para o resumo do cenário
                gpt_payload = {
                    "scenario": scenario, "desc": SCENARIO_DESC.get(scenario, ""),
                    "profile": perfil_nome,
                    "settings": dict(concurrency=perfil["concurrency"], requests=perfil["requests"],
                                     temperature=perfil["temperature"], max_tokens=perfil["max_tokens"],
                                     model_key=model_key),
                    "execution_time": {"sdk_seconds": time_sdk, "proxy_seconds": time_proxy},
                    "sdk": {"ok": r_sdk["ok"], "errors": r_sdk["errors"], "p50": r_sdk["p50"], "p95": r_sdk["p95"],
                            "ttft_p50": r_sdk["ttft"], "throughput": thr_sdk},
                    "proxy": {"ok": r_proxy["ok"], "errors": r_proxy["errors"], "p50": r_proxy["p50"], "p95": r_proxy["p95"],
                              "ttft_p50": r_proxy["ttft"], "throughput": thr_proxy},
                    "derived": derived,
                    "winners": dict(p50=win_p50, p95=win_p95, throughput=win_thr)
                }
                render_practical_gpt_summary(scenario_name=scenario, scenario_desc=SCENARIO_DESC.get(scenario, ""),
                                             data=gpt_payload, meta=run_meta)

                # Acumula para o resumo executivo final
                overall_results.append({
                    "scenario": scenario, "desc": SCENARIO_DESC.get(scenario, ""),
                    "settings": dict(concurrency=perfil["concurrency"], requests=perfil["requests"],
                                     temperature=perfil["temperature"], max_tokens=perfil["max_tokens"], model_key=model_key),
                    "execution_time": {"sdk_seconds": time_sdk, "proxy_seconds": time_proxy},
                    "sdk": dict(ok=r_sdk["ok"], errors=r_sdk["errors"], p50=r_sdk["p50"], p95=r_sdk["p95"], ttft_p50=r_sdk["ttft"], throughput=thr_sdk),
                    "proxy": dict(ok=r_proxy["ok"], errors=r_proxy["errors"], p50=r_proxy["p50"], p95=r_proxy["p95"], ttft_p50=r_proxy["ttft"], throughput=thr_proxy),
                    "derived": derived, "winners": dict(p50=win_p50, p95=win_p95, throughput=win_thr)
                })

        if overall_results:
            st.divider()
            render_technical_gpt_overall_summary(overall_results, run_meta)
