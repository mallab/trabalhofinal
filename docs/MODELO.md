# Modelo de recomendação e decisões do projeto

## 1. Ideia geral

O sistema usa um **modelo híbrido ponderado**. São três recomendadores separados,
cada um dá uma nota de 0 a 1 pra cada filme, e no final a gente soma essas notas
com pesos que **mudam conforme o histórico do usuário**:

```
nota_final = peso_cf × COLABORATIVA + peso_conteudo × CONTEÚDO + peso_pop × POPULARIDADE
```

Depois de somar, tiramos os filmes que o usuário já avaliou e os dos gêneros que
ele marcou como "não gosto", pegamos os N melhores e geramos uma frase explicando
o motivo de cada um.

## 2. As três partes

### 2.1 Filtragem colaborativa (SVD com fold-in)

1. Montamos uma matriz usuário × filme com as notas. Antes, tiramos a média de
   cada usuário das notas dele, porque tem gente que dá 5 pra tudo e gente que é
   mais exigente. Pra quem tem poucas notas, a média é puxada pra média geral:
   `(soma + 5 × média_geral) / (n + 5)`.
2. Aplicamos o `TruncatedSVD` do scikit-learn com **20 dimensões**. Isso resume
   o gosto de cada filme em 20 números (a matriz `Vt`).
3. **Fold-in**: pra recomendar, pegamos o vetor de notas do usuário, projetamos
   nessas 20 dimensões e voltamos pro tamanho do catálogo:

   ```
   p = Vt · notas        (o gosto do usuário em 20 números)
   previsao = Vtᵀ · p    (nota prevista pra cada filme)
   ```

**Por que fold-in?** Assim não precisa guardar um vetor por usuário nem retreinar
quando alguém dá nota. A projeção usa as notas atuais, lidas direto do banco.
Usuário criado depois do treino já recebe recomendação colaborativa. Isso faz o
endpoint de "atualizar preferências" ter efeito na hora.

**Por que SVD e não KNN de usuários?** A matriz do MovieLens é 98,3% vazia
(610 usuários × 9.742 filmes). Comparar usuários direto fica muito ruidoso, e os
fatores do SVD generalizam melhor. Recomendar vira só uma multiplicação de
matrizes, que leva milissegundos.

**Limite de dimensões:** o número de dimensões fica limitado a 1/8 da menor
dimensão da matriz. Percebemos isso nos testes: com a base pequena (32
usuários), 20 dimensões faziam o SVD "decorar" a matriz e as recomendações
saíam erradas. No MovieLens esse limite dá 76, então continua valendo o 20.

### 2.2 Filtragem por conteúdo (TF-IDF)

- Cada filme vira uma lista com os **gêneros** dele + a **década**
  (ex: `decada_1990`), que tiramos do título. A década ajuda a pegar gostos tipo
  "filmes dos anos 80", que só o gênero não pega.
- O TF-IDF dá menos peso pra gênero muito comum (Drama) e mais pra gênero raro
  (Film-Noir).
- **Perfil do usuário** = soma dos filmes que ele avaliou × (nota − 3), mais
  2 × os gêneros favoritos que ele escolheu. Nota abaixo de 3 empurra o perfil
  pra longe daquele estilo de filme.
- A nota de cada filme é o cosseno entre o perfil e o filme.

Isso resolve o **cold start de filme**: um filme recém-cadastrado, sem nenhuma
nota, já tem gêneros e pode ser recomendado.

### 2.3 Popularidade (média bayesiana)

```
nota_bayes = (n × média + C × média_geral) / (n + C)       com C = 10
POPULARIDADE = 0,7 × nota_bayes (normalizada) + 0,3 × log(1 + n) (normalizado)
```

A média bayesiana é a mesma fórmula do ranking do IMDb. Ela evita que um filme
com uma nota 5 só vá pro topo. Juntamos com a quantidade de notas porque, sem
isso, os filmes sem nenhuma nota ficavam com nota "média" e apareciam na frente
de filmes muito bons. Com essa mudança, a precisão da estratégia `popular` subiu
de 0,078 pra 0,108.

## 3. Pesos que mudam com o histórico

Chamando de `m = min(1, quantidade_de_notas / 20)` o quanto o usuário já é
"conhecido":

| Parte | Peso |
|---|---|
| Colaborativa | `0,7 × m` |
| Popularidade | `0,1 + 0,2 × (1 − m)` |
| Conteúdo | o que sobrar |

| Situação | Colab. | Conteúdo | Pop. |
|---|---:|---:|---:|
| Usuário novo sem gêneros favoritos | 0 | 0 | 1,00 |
| Usuário novo com gêneros favoritos | 0 | 0,70 | 0,30 |
| 1 nota | 0,035 | 0,675 | 0,29 |
| 7 notas | 0,245 | 0,525 | 0,23 |
| 20 notas ou mais | 0,70 | 0,20 | 0,10 |

**Por quê:** com pouca nota a colaborativa erra muito, então é mais seguro usar
os gêneros com uma ajuda da popularidade (pra não sugerir filme obscuro). Quando
o usuário já tem bastante nota, a colaborativa, que foi a melhor parte sozinha
na avaliação, passa a mandar. Se não tiver informação nenhuma, cai pra
popularidade e a API avisa isso no campo `strategy_used`.

## 4. Filtros e explicações

- Filme que o usuário já avaliou e filme de gênero que ele marcou como "não
  gosto" **nunca** aparecem. Preferimos tirar de vez em vez de só diminuir a
  nota, porque o usuário pediu pra não ver.
- Cada recomendação mostra quanto cada parte contribuiu (`components`) e uma
  frase baseada na parte que mais pesou:
  - colaborativa: _"Quem curtiu 'X' também curtiu este filme"_
  - conteúdo: _"Parecido com 'X', que você avaliou bem"_ ou
    _"Combina com seus gêneros favoritos: ..."_
  - popularidade: _"Popular e bem avaliado no catálogo (317 avaliações, média 4.4)"_

## 5. Quando o modelo é retreinado

| O que aconteceu | O que o sistema faz |
|---|---|
| Nota nova ou alterada | Nada na hora, o fold-in já usa as notas atuais. Depois de 1000 notas novas, marca pra retreinar. |
| Filme novo | Marca pra retreinar e retreina na próxima consulta (uns 0,3 s). |
| `POST /model/retrain` | Retreina na hora. |

Treinar o MovieLens small inteiro leva uns 0,3 segundos, então dá pra fazer em
memória. Usamos um lock pra duas requisições não retreinarem ao mesmo tempo.

## 6. Avaliação e escolha dos parâmetros

Como avaliamos (`app/evaluation.py`): pra cada usuário com pelo menos 20 notas,
escondemos 20% delas e treinamos com o resto. Conta como acerto quando um filme
escondido que o usuário avaliou com nota 4 ou mais aparece no top 10.

Testamos combinações de parâmetros na estratégia híbrida: dimensões do SVD
(20, 50 e 100), peso máximo da colaborativa (0,6, 0,7 e 0,8) e peso da
popularidade (0,1 e 0,2). A diferença foi pequena. Ficamos com 20 dimensões e
colaborativa 0,7, que teve boa precisão e a melhor cobertura.

Resultado final (595 usuários):

| Estratégia | Precision@10 | Recall@10 | Hit rate@10 | Cobertura |
|---|---:|---:|---:|---:|
| **hybrid** | **0,154** | **0,134** | **63,9%** | 5,5% |
| collaborative | 0,137 | 0,118 | 59,5% | 4,8% |
| content | 0,031 | 0,027 | 22,0% | 18,9% |
| popular | 0,114 | 0,098 | 54,0% | 0,4% |

O conteúdo sozinho é fraco no MovieLens porque o único dado que temos dos filmes
é o gênero. Mas ele é o que mais varia os filmes recomendados (maior cobertura)
e é essencial pra usuário novo. Junto com a colaborativa, melhora o resultado.

## 7. Decisões de arquitetura

| Decisão | Motivo |
|---|---|
| **FastAPI + Pydantic** | Valida os dados sozinho (nota entre 0,5 e 5, de meio em meio, senão dá erro 422) e gera o Swagger automaticamente. |
| **SQLAlchemy** com SQLite ou PostgreSQL | O mesmo código funciona nos dois. SQLite pra rodar local e nos testes sem instalar nada, PostgreSQL no Docker. |
| **Modelo separado da API** | O `recommender.py` não sabe nada de banco nem de HTTP, então dá pra testar sozinho e usar no script de avaliação. |
| **Camada de serviço** (`service.py`) | Concentra a regra de quando retreinar e o controle de concorrência. |
| **Gêneros salvos como `A\|B\|C`** | Mesmo formato do MovieLens. A API mostra como lista e corrige maiúsculas (`sci-fi` vira `Sci-Fi`). |
| **Carga do banco só uma vez** | A API pode reiniciar sem duplicar dados. No Postgres ajustamos o contador de id depois de importar os ids do MovieLens. |
| **Dataset baixado no build** | O repositório fica leve. Sem internet, usa a base pequena de `data/sample`. |
| **Container sem root e com healthcheck** | Mais seguro, e o Compose só sobe a API depois que o Postgres está pronto. |

## 8. Limitações e o que dá pra melhorar

- O único dado de conteúdo é gênero e ano. Usar o `tags.csv` do MovieLens ou a
  sinopse dos filmes deve melhorar bastante essa parte.
- A cobertura baixa (uns 5%) é normal em filtragem colaborativa. Dá pra aumentar
  reordenando a lista pra ter mais variedade (MMR).
- O retreino acontece dentro da requisição. Num sistema de verdade iria pra uma
  tarefa em segundo plano.
- O modelo fica na memória de um processo só. Com vários workers, cada um teria
  a sua cópia, o que é tranquilo pro tamanho do MovieLens small.
