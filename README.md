# benchmark-litellm-sdk-proxy

Benchmark que compara duas formas de chamar um LLM pelo LiteLLM: o SDK dentro do processo da aplicação e o LiteLLM Proxy (gateway HTTP compatível com a API da OpenAI). Mede latência, tempo até o primeiro token (TTFT), taxa de erro e custo estimado em três cenários típicos de chatbot, e mostra o resultado num painel Streamlit.

## Por que existe

Antes de colocar um gateway de LLM na frente de vários bots, eu queria saber quanto ele custa em latência e estabilidade comparado à chamada direta. Em vez de opinião, um teste de carga com os mesmos prompts nos dois caminhos.

## O que mede

| cenário | o que exercita |
|---|---|
| `chat` | resposta completa, sem streaming e sem tools |
| `stream` | streaming, com medição de TTFT |
| `tools` | resposta com definição de tools no payload |

- `run_bench.py` dispara as requisições com concorrência configurável e grava `reports/results_*.csv` linha a linha e `reports/summary_*.md` com percentis e custo (preços em `pricing.yaml`, USD por 1 mil tokens).
- `app.py` (Streamlit) roda os cenários, mostra placares, curvas ECDF de latência, progresso com ETA e um resumo curto gerado por LLM.
- `summarize.py` consolida os relatórios mais recentes no terminal.

## Stack

Python 3.11+, LiteLLM (SDK e Proxy em Docker), OpenAI SDK, Streamlit, Plotly, pandas.

## Como rodar

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                                   # preencha OPENAI_API_KEY e LITELLM_PROXY_KEY

docker compose -f docker-compose.proxy.yaml up -d      # proxy em http://localhost:4000
streamlit run app.py                                   # painel em http://localhost:8501
```

Sem o painel:

```bash
python run_bench.py --mode sdk   --scenario stream --requests 200 --concurrency 20
python run_bench.py --mode proxy --scenario stream --requests 200 --concurrency 20
python summarize.py
```

Os modelos expostos pelo proxy ficam em `proxy_config.yaml`. Os prompts de cada cenário estão em `scenarios/`.

## Testes

Não há testes automatizados; o próprio repositório é uma ferramenta de medição. Rodar exige chave de API e gera custo de tokens.

## Status

Ferramenta de estudo, usada para uma comparação pontual. Os números dependem do provedor, do modelo e da rede, por isso nenhum resultado fica versionado (`reports/` está no `.gitignore`).

## Licença

MIT.
