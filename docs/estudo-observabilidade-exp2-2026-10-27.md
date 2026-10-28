# Estudo — Issue #8 (parte 2): Experimento OBS-2 — histórico de observações

**Pergunta:** se o gargalo for (também) temporal — o frame único não distingue
"tocou" de "segurando", nem captura de repouso — qual é a **menor mudança** que
dá memória à política, e como testá-la sem confundir com o problema da
interface BC→PPO?
**Relação com a parte 1** (`docs/estudo-observabilidade-2026-10-27.md`): aquele
estudo ranqueou as hipóteses e pré-registrou o **Experimento 1** (diagnóstico
em 2 estágios + warm-start protegido). Este documento (1) consolida o estado
das evidências, (2) separa de novo as hipóteses com seus discriminadores,
(3) desenha o **Experimento 2 (OBS-2)** com protocolo pré-registrado, e
(4) entrega a mudança mínima **já implementada atrás de flag** nesta branch.

---

## 0. Estado da frente (27/10/2026, fim do dia)

- **Experimento 1**: implementado na main (Estágio A `eval_bc_actor.py`;
  Estágio B `--protected` no `train_ppo_bc.py`, com a correção do off-by-one
  do `learn()`); kernel v8 lançado sem a correção foi **invalidado** e
  substituído pela **v9** ainda no estágio de coleta. **Veredito ainda não
  registrado na SPEC** — a consulta de status do kernel
  `fanuc-bc-warmstart` via API retornou 403 (`kernels.get` negado) nesta
  sessão, então o estado da corrida v9 não é afirmado aqui.
- **OBS-2 (este documento)**: desenho + protocolo abaixo; a fiação da
  mudança mínima (frame stacking) está implementada e testada nesta
  branch. A corrida em si **só roda quando a árvore de decisão da §2
  mandar** — empilhar o OBS-2 sobre o Exp. 1 sem ler o veredito dele
  destruiria o poder de bisseção do desenho em 2 estágios.

## 1. Evidências consolidadas: o que a política vê, e o que continua invisível

Observação atual da política (**24 dims**, grupo `policy` em
`config/m10id12/env_cfg.py`):

| Termo | Dims | Fonte |
|---|---|---|
| `joint_position` (rel.) | 8 | J1–J6 + 2 dedos da garra |
| `joint_velocity` (rel.) | 8 | idem |
| `part_relative_to_tcp` | 3 | posição da peça no referencial do TCP |
| `part_velocity` | 3 | velocidade linear da peça (mundo) |
| `finger_contact` | 2 | força de contato de cada dedo (v68, issue #1) |

Controle: `dt=0,01` × decimação 2 → **passo de política de 0,02 s (50 Hz)**;
episódio de 12 s = **600 passos**. Fatos que enquadram qualquer explicação
(registros da SPEC e da parte 1): professor scriptado **0,865** (age por
fases, com latch de captura); PPO puro **0,0** em 9,2M amostras (v70);
BC convergido (loss 0,000157) → PPO warm-start com **grasp 0,0 desde a
iteração inicial** e benchmark **0,0** (BC v7).

O que um frame continua **não** contendo (síntese da parte 1, §H2):

| Variável latente | Por que é invisível num frame | Escala temporal |
|---|---|---|
| Captura (peça travada à garra × repouso na esteira) | vel. ≈ 0 e posição rel. parecidas nos dois casos; a força instantânea é transitória | sub-segundo |
| Dinâmica do contato (pico entre passos de controle) | força amostrada a 50 Hz pode perder o evento de 100 Hz da simulação | 10–100 ms |
| Escorregamento durante o lift | só aparece integrado no tempo | 0,1–1 s |
| Fase/tentativa (insistir × abortar após grasp falho) | estados geometricamente iguais pedem ações opostas | segundos |
| Velocidade da esteira após a captura | a vel. da peça deixa de refleti-la | episódio |

Nota de leitura: o contato **instantâneo** entrou na observação na v68 e o
zero persistiu na v7 — a hipótese temporal não é "falta contato", é "falta
a **dinâmica** do contato e do estado de captura", que só existe ao longo
de uma janela.

## 2. Hipóteses concorrentes e o que as distingue empiricamente

| Hipótese | Previsão falseável | Quem a testa |
|---|---|---|
| **H1 — destruição do warm-start na interface BC→PPO** (ruído/entropia/crítico frio/normalização/sem âncora) | ator BC puro pega em rollout; grasp@iteração-0 do PPO ≈ 0 mesmo protegido | **Exp. 1** (em voo) |
| **H2 — aliasing temporal** (captura/contato/fase não identificáveis num frame) | BC puro falha em rollout **mesmo com obs perfeita de frame**; BC com história melhora o rollout | Exp. 1 Estágio A (lado BC) + **OBS-2** |
| **H3 — crítico simétrico fraco** (vantagens ruidosas em estados aliasados) | treino instável/lento mas **não** zero absoluto desde o início | co-fator; OBS-3 (crítico assimétrico) |
| **H4 — BC não transfere nem sozinho** (covariate shift) | `grasp_rate_BC < 0,05` no Estágio A | Exp. 1 Estágio A; remédio canônico = DAgger |

**Árvore de decisão do OBS-2 (pré-registrada, lê o veredito do Exp. 1):**

- Exp. 1 **PROVA** → rota warm-start validada; OBS-2 vira estudo de
  robustez/teto (história ajuda além de 0,30?), prioridade do fundador.
- Exp. 1 **PARCIAL (ii)** (grasp ≥ 0,20 sem sucesso) → o gargalo migrou
  para **gerenciar** o grasp (lift/place) — exatamente o território do
  aliasing de captura/escorregamento → **OBS-2 indicado como próximo**.
- Exp. 1 **PARCIAL (iii)** ou **MATA-A** (grasp nasce e decai/colapsa no
  PPO) → esquecimento durante o PPO confirmado → próximo é **DAPG/KL**,
  **não** OBS-2 (história não protege peso clonado).
- Exp. 1 **MATA-B** (`grasp_rate_BC < 0,05` já no Estágio A) → o BC em si
  não expressa o grasp → **OBS-2 vira a frente principal**: o Estágio A
  repetido sob história mede diretamente se o aliasing temporal era o
  limitante do BC.

## 3. Opções de mudança mínima (custo × benefício)

Controle a 50 Hz: H frames cobrem H×0,02 s de passado.

| Opção | Mudança | Custo | O que compra | O que NÃO compra |
|---|---|---|---|---|
| **(a) Frame stacking H=8** (0,16 s; 192 dims) | gate no grupo de obs + flag no builder (**implementada nesta branch**) | ~zero de engenharia nova; re-coleta obrigatória (demos mudam de dim) | dinâmica do contato, latch de captura, tendência da peça; MLP e pipeline BC intactos (só a camada de entrada cresce) | memória de tentativa (segundos atrás segue invisível — nenhum H prático cobre 600 passos) |
| (b) Features derivadas de sensores | termos MDP novos: tempo normalizado do episódio, vel. comandada da esteira, flag "peça assegurada" (garra fechada + força bilateral sustentada), ação anterior | médio: `mdp/cell.py` + exports (lição da v68b), re-coleta; cada feature pede a análise de legitimidade de deploy da parte 1 (§5) | ataca captura/fase **semanticamente**, com poucas dims | é desenho manual de estado — vieses do projetista; não cobre transientes |
| (c) Crítico assimétrico privilegiado | grupo `critic` (vel. real da esteira, estado de captura, fase) + `obs_groups` no cfg do agente | médio-baixo; ator e BC intactos | vantagens melhores (H3); compõe com (a)/(b) | ator continua cego na decisão |
| (d) Ator recorrente (RNNModel GRU/LSTM) | `RslRlRNNModelCfg` + BC **por sequências** + hidden state no runner | alto: pré-treino do BC muda de natureza (pares i.i.d. → sequências), treino mais lento/delicado | memória de horizonte arbitrário (tentativa falha há segundos) | tudo o que (a) compra, a custo muito maior; só se justifica com evidência de dependência longa |

**Recomendação: (a) agora** — é a menor mudança que ataca o aliasing
temporal *mensurável* (sub-segundo: contato e captura), usa o mecanismo
nativo do Isaac Lab, não toca loop de treino, MDP ou reward, e compõe
com (b)/(c) depois. (d) fica desenhada para o caso de a evidência
apontar dependência de segundos (tentativas), que stacking não cobre
por construção — registrar esse limite evita a leitura errada de um
MATA do OBS-2 como "memória não importa".

## 4. Protocolo pré-registrado — Experimento OBS-2 (Kaggle)

**Nome:** OBS-2 — "História de 8 frames muda o BC e o grasp?"
**Pré-condição:** veredito do Exp. 1 lido e ramo da árvore da §2
apontando para o OBS-2 (PARCIAL-ii ou MATA-B; ou decisão do fundador
no ramo PROVA). Kernel novo, slug próprio `fanuc-obs2-history`
(build com `--bc-warmstart --bc-exp1 --obs-history 8
--train-num-envs 256 --train-max-iterations 1500`); nada roda na VM.

**Única diferença vs. o Experimento 1:** `FANUC_CELL_OBS_HISTORY=8`
— os últimos 8 frames (0,16 s) de todos os termos do grupo `policy`,
achatados (24 → 192 dims). Todo o resto é herdado **por construção**
(mesmo notebook, mesmas flags): coleta do professor idêntica (portão
≥ 0,80), BC idêntico (300 épocas, demos frescas coletadas já com a
obs empilhada), Estágio A idêntico (ator BC puro, 200 episódios,
seeds 2000+), Estágio B idêntico (PPO protegido: entropia 0, std
inicial 0,1, crítico aquecido com ator congelado por 50 iterações),
budget 256×1500 (~9,2M amostras), **benchmark multi-seed B3** (101,
202, 303; média ± desvio).

**Desvios forçados, registrados:** (i) a camada de entrada do MLP
[256,128,64] cresce de 24 para 192 entradas — inevitável em qualquer
mudança de observação, não é grau de liberdade do experimento;
(ii) demos e checkpoints desta corrida **não** são comparáveis nem
intercambiáveis com os de H=1 (dimensão diferente) — a comparação
válida é entre **métricas de rollout/benchmark**, nunca entre pesos.

**Métricas oficiais:** `grasp_rate_BC(H=8)` e `success_rate_BC(H=8)`
do Estágio A (comparáveis diretamente aos do Exp. 1);
`grasp@iteração-0`; benchmark `success_rate` (primária), `grasp_rate`
final e máximo; sucesso do professor na coleta; `obs_history=8`
presente no runtime-report (o builder o injeta — ausência invalida a
corrida como teste de H2).

**Portões (contra o 0,0 histórico e contra o veredito do Exp. 1):**
- **PROVA:** benchmark `success_rate ≥ 0,30` ⇒ memória curta entra
  para a receita padrão; próximo passo vira escala/robustez.
- **PARCIAL:** qualquer um de —
  (i) `0 < success_rate < 0,30`;
  (ii) `success_rate = 0` mas `grasp_rate` final ≥ 0,20;
  (iii) `grasp_rate_BC(H=8) ≥ 0,30` com colapso no PPO (o aliasing
  limitava o **BC** e foi tratado; a interface segue quebrada →
  próximo experimento = DAPG **sobre** a obs com história).
- **MATA:** `success_rate = 0` **e** `grasp_rate` final ≤ 0,02 **e**
  `grasp_rate_BC(H=8) < 0,05` ⇒ aliasing temporal de janela curta
  (0,16 s) rejeitado como causa primária na escala testada; próximos
  candidatos voltam à decisão do fundador com os dados: features
  derivadas (opção b), crítico assimétrico (c) ou recorrência (d).

**Integridade:** seeds e protocolo de avaliação idênticos aos do
Exp. 1; qualquer diferença além do item pré-registrado (história=8 e
seus dois desvios forçados) invalida a corrida como teste de H2 e
deve ser registrada como tal no relatório.

## 5. Implementado nesta branch × só desenhado

**Implementado (com testes estruturais/funcionais, suíte verde):**
- `env_cfg.py`: helper `_obs_history_length()` (lê
  `FANUC_CELL_OBS_HISTORY`, default 1, valida ≥ 1 com erro alto) e o
  gate no `PolicyCfg.__post_init__` — com 1, **nenhum** atributo de
  história é tocado (`history_length` segue `None`, sem buffers, sem
  mudança de dimensão: default byte a byte o de sempre); com N ≥ 2,
  `history_length = N` + `flatten_history_dim = True` no grupo.
- `scripts/build_kaggle_notebook.py`: flag `--obs-history N`
  (default 1, validada) que injeta a variável no **processo do
  notebook** (herdado pelo `isaaclab train`, que não recebe
  `baseline_env`) e no **`baseline_env`** (coleta, eval do BC,
  baseline), mais o marcador `runtime_report['obs_history']`.
  Sem a flag, o notebook gerado não contém nenhum traço da variável
  (testado construindo os dois notebooks).
- `tests/test_obs_history.py`: 6 testes (gate estrutural, helper
  funcional extraído por AST, flag/validação do builder, notebook
  default limpo, injeção completa com N=8).

**Só desenhado (não implementado):** as opções (b) features derivadas,
(c) crítico assimétrico e (d) ator recorrente da §3 — cada uma é um
experimento próprio (OBS-3+) e depende do veredito que a árvore da §2
produzir. A corrida OBS-2 em si é Kaggle e aguarda a pré-condição do
protocolo (veredito do Exp. 1).

## Referências

- Parte 1 deste estudo (hipóteses, DAPG/FLaRe, legitimidade de
  features, RNN no RSL-RL): `docs/estudo-observabilidade-2026-10-27.md`.
- Isaac Lab — API `isaaclab.managers`: `ObservationGroupCfg`
  (`history_length: int | None` sobrescreve o dos termos;
  `flatten_history_dim` default True).
  https://isaac-sim.github.io/IsaacLab/v2.3.1/source/api/lab/isaaclab.managers.html
- Hausknecht & Stone 2015 (DRQN) e Heess et al. 2015 — em POMDP a
  política ótima é função da história, não do frame (via parte 1).
- Protocolo do Experimento 1 e vereditos herdados: `SPEC.md`.
