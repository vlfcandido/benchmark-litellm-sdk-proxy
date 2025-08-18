# LiteLLM — Benchmark SDK vs Proxy

Benchmark em Streamlit para comparar LiteLLM SDK e LiteLLM Proxy em cenários de chatbot/ADK.

## Instalação

```bash
# Clone o projeto
git clone https://github.com/seu-repo/litellm-bench.git
cd litellm-bench

# Crie um ambiente virtual (recomendado)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Instale dependências
pip install -U pip
pip install -r requirements.txt
```

## Configuração

Crie um arquivo `.env` na raiz do projeto:

```dotenv
# SDK
OPENAI_API_KEY=sua_chave_openai
OPENAI_BASE_URL=https://api.openai.com/v1
LITELLM_SDK_MODEL=openai/gpt-4o-mini

# Proxy
LITELLM_PROXY_URL=http://localhost:4000/v1
LITELLM_PROXY_KEY=sk-proxy-123

# Resumos GPT
SUMMARY_MODEL=gpt-4o-mini
```

E adicione um arquivo `proxy_config.yaml` com os provedores/modelos que deseja expor pelo proxy.  
Exemplo básico:

```yaml
model_list:
  - model_name: gpt-4o-mini
    litellm_params:
      model: gpt-4o-mini
      api_key: ${OPENAI_API_KEY}
```

## Executando

### 1. Subir o LiteLLM Proxy com Docker Compose
```bash
docker compose -f docker-compose.proxy.yaml up -d
```

O proxy ficará disponível em `http://localhost:4000/v1`.

### 2. Rodar o app Streamlit
```bash
streamlit run app.py
```

O app abrirá no navegador em `http://localhost:8501`.

## Estrutura mínima

```
.
├── app.py                      # UI principal (Streamlit)
├── proxy_config.yaml           # Configuração do LiteLLM Proxy
├── docker-compose.proxy.yaml   # Subida do proxy
├── .env.example                # Exemplo de variáveis de ambiente
└── requirements.txt            # Dependências do projeto
```

## Passos finais

1. Configure `.env`  
2. Suba o proxy com `docker compose -f docker-compose.proxy.yaml up -d`  
3. Rode `streamlit run app.py`  

O dashboard exibirá KPIs e comparações SDK vs Proxy.
