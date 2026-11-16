# Sistema de Recomendação de Filmes (FastAPI + Docker)

Trabalho de Desenvolvimento de Sistemas: API de recomendação de filmes usando o
dataset **MovieLens small** (100.836 notas, 9.742 filmes e 610 usuários).

O modelo é **híbrido**: junta filtragem colaborativa, filtragem por conteúdo e
popularidade. A explicação completa do modelo e das decisões que tomamos está em
[docs/MODELO.md](docs/MODELO.md).

**Autores:** _preencher com nome e matrícula_

---

## O que o projeto faz

- Recomenda filmes juntando 3 técnicas, e o peso de cada uma muda conforme o
  usuário tem mais ou menos notas
- Quando o usuário dá uma nota, a recomendação muda na hora, sem retreinar o modelo
  (usamos fold-in no SVD)
- Usuário novo recebe recomendação pelos gêneros favoritos + popularidade, e filme
  novo sem nenhuma nota também pode ser recomendado (pelos gêneros)
- Cada recomendação vem com o motivo em português, tipo
  _"Parecido com 'Matrix', que você avaliou bem"_
- Dá pra escolher entre 4 estratégias: `hybrid`, `collaborative`, `content` e `popular`
- Tem uma avaliação offline comparando as 4 (precision@10, recall@10, cobertura)
- 59 testes automáticos (unitários e de integração)
- Docker Compose sobe a API e um PostgreSQL juntos

## Estrutura

```
app/
  main.py          cria a API e o Swagger; na subida carrega os dados e treina
  recommender.py   o modelo híbrido (SVD + TF-IDF + popularidade)
  service.py       liga o modelo com o banco e decide quando retreinar
  evaluation.py    avaliação offline
  seed.py          baixa o MovieLens e carrega no banco
  models.py        tabelas: users, items, ratings
  schemas.py       formato dos dados que entram e saem da API
  crud.py          funções de banco
  database.py      conexão (SQLite local ou PostgreSQL no Docker)
  config.py        configurações
  routers/         rotas: users.py, items.py, system.py
tests/             testes do modelo, da API e da carga de dados
scripts/           scripts de instalar, rodar, testar, baixar dados e avaliar
data/sample/       base pequena usada nos testes
docs/MODELO.md     explicação do modelo
Dockerfile
docker-compose.yml
```

## Como rodar

### Com Docker (recomendado)

Precisa do Docker Desktop aberto.

```bash
docker compose up --build
```

Quando terminar de subir, abre **http://localhost:8000/docs**. Na primeira vez a
API baixa o MovieLens, carrega no Postgres e treina o modelo (uns 10 segundos).

Pra parar: `Ctrl + C` e depois `docker compose down`
(com `docker compose down -v` apaga também os dados do banco).

### Sem Docker (local, com SQLite)

Precisa do Python 3.11 ou mais novo.

| | Mac / Linux | Windows (PowerShell) |
|---|---|---|
| Instalar | `./scripts/install.sh` | `.\scripts\install.ps1` |
| Rodar | `./scripts/run.sh` | `.\scripts\run.ps1` |
| Testes | `./scripts/test.sh` | `.\scripts\test.ps1` |

Ou na mão:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python -m scripts.download_data    # opcional, sem ele usa a base pequena
uvicorn app.main:app --reload
```

### Variáveis de ambiente

| Variável | Padrão | Pra que serve |
|---|---|---|
| `DATABASE_URL` | `sqlite:///recsys.db` | No Docker vira o PostgreSQL |
| `AUTO_SEED` | `true` | Carrega o MovieLens na primeira vez |
| `DOWNLOAD_DATA` | `true` | Baixa o MovieLens se ele não estiver em `data/` |
| `DATA_DIR` | `./data` | Pasta dos CSVs |
| `SVD_COMPONENTS` | `20` | Número de dimensões do SVD |
| `CF_FULL_WEIGHT_AT` | `20` | Com quantas notas a colaborativa chega no peso máximo |
| `POPULARITY_PRIOR` | `10` | Constante da média bayesiana |

## Endpoints

A documentação interativa (Swagger) fica em **`/docs`**.

| Método | Rota | O que faz |
|---|---|---|
| `POST` | `/users` | **Adiciona usuário** (pode mandar gêneros favoritos e os que não gosta) |
| `GET` | `/users/{id}` | Consulta usuário |
| `PUT` | `/users/{id}/preferences` | **Atualiza preferências** de gênero |
| `POST` | `/users/{id}/ratings` | **Atualiza preferências** dando nota (cria ou atualiza) |
| `GET` | `/users/{id}/ratings` | Lista as notas do usuário |
| `DELETE` | `/users/{id}/ratings/{item_id}` | Remove uma nota |
| `GET` | `/users/{id}/recommendations?n=10&strategy=hybrid` | **Recomendações** pro usuário |
| `POST` | `/items` | **Adiciona item** (filme) |
| `GET` | `/items?search=&genre=` | Busca filmes |
| `GET` | `/items/{id}` | Detalhes do filme (quantidade de notas e média) |
| `GET` | `/items/{id}/similar` | Filmes parecidos |
| `GET` | `/genres` | Lista de gêneros |
| `GET` | `/health` | Confere se a API e o banco estão funcionando |
| `GET` | `/model/status` | Versão e estado do modelo |
| `POST` | `/model/retrain` | Retreina o modelo na hora |

### Exemplo

```bash
# 1. cria o usuário
curl -X POST localhost:8000/users -H "Content-Type: application/json" \
  -d '{"name": "Ana", "preferred_genres": ["Sci-Fi", "Action"], "disliked_genres": ["Horror"]}'

# 2. dá nota pra um filme (2571 = Matrix)
curl -X POST localhost:8000/users/611/ratings -H "Content-Type: application/json" \
  -d '{"item_id": 2571, "rating": 5}'

# 3. pede as recomendações
curl "localhost:8000/users/611/recommendations?n=5"
```

Resposta (resumida) depois de dar 5 pro Matrix:

```json
{
  "user_id": 611,
  "strategy_used": "hybrid",
  "weights": {"collaborative": 0.035, "content": 0.675, "popular": 0.29},
  "recommendations": [
    {
      "item": {"id": 589, "title": "Terminator 2: Judgment Day (1991)",
               "genres": ["Action", "Sci-Fi"], "year": 1991},
      "score": 0.9284,
      "reason": "Parecido com 'Matrix, The (1999)', que você avaliou bem",
      "components": {"collaborative": 0.018, "content": 0.6688, "popular": 0.2416}
    }
  ]
}
```

Com só 1 nota a colaborativa quase não pesa (0.035). Conforme o usuário avalia
mais filmes, esse peso vai subindo até 0.7.

## Testes

```bash
pytest -v
```

- `tests/test_recommender.py`: testes **unitários** do modelo (cold start, filtros,
  pesos, filmes parecidos, avaliação)
- `tests/test_api.py`: testes de **integração**, sobe a API com banco de verdade e
  testa todos os endpoints, os erros (404 e 422) e o fluxo completo
  cadastrar → avaliar → recomendar
- `tests/test_seed.py`: carga dos dados no banco

Os testes usam a base pequena de `data/sample` (40 filmes e 32 usuários), então
rodam em menos de 1 segundo e sem internet.

Dentro do Docker: `docker compose run --rm api pytest -v`

## Avaliação offline

```bash
python -m scripts.evaluate --users 0
```

Pra cada usuário com pelo menos 20 notas escondemos 20% delas e vemos se o modelo
recomenda os filmes escondidos que ele avaliou com nota 4 ou mais.

Resultado no MovieLens small (595 usuários):

| Estratégia | Precision@10 | Recall@10 | Hit rate@10 | Cobertura |
|---|---:|---:|---:|---:|
| **hybrid** | **0.154** | **0.134** | **63.9%** | 5.5% |
| collaborative | 0.137 | 0.118 | 59.5% | 4.8% |
| content | 0.031 | 0.027 | 22.0% | 18.9% |
| popular | 0.114 | 0.098 | 54.0% | 0.4% |

O híbrido acertou 12% mais que a colaborativa sozinha e 34% mais que a
popularidade. Os números podem variar um pouco de máquina pra máquina.

## Dados

F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.*
ACM TiiS 5, 4. https://grouplens.org/datasets/movielens/

O dataset não fica no repositório: ele é baixado pelo `scripts/download_data.py`
ou durante o `docker build`.
