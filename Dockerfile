# Imagem da API de recomendação
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# instala as bibliotecas antes de copiar o código, assim quando só o código
# muda o Docker não precisa instalar tudo de novo
COPY requirements.txt requirements-dev.txt ./
RUN pip install -r requirements-dev.txt

COPY app ./app
COPY scripts ./scripts
COPY tests ./tests
COPY data ./data
COPY pytest.ini ./

# baixa o MovieLens no build. Se não tiver internet usa a base pequena
# de data/sample (o build não quebra)
RUN python -m scripts.download_data || echo "Aviso: MovieLens não baixado; será usada a amostra."

# roda com um usuário comum em vez de root (mais seguro)
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# a cada 15s confere se a API está respondendo
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
