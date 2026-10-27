# Pendências para o fundador — sessão noturna BC (27/10/2026)

O fundador dormindo não aprova os cartões do conector GitHub; por
regra dele, esta noite toda escrita no GitHub foi por git normal.
O item abaixo ficou pronto para colar, sem perda.

## Comentário pronto para a issue #3 (BC warm-start)

> ## Preparação do experimento BC mergeável (27/10/2026, sessão noturna)
>
> A issue inteira foi preparada em código na branch
> `frente/bc-warmstart-prep` (sem merge — aguarda revisão), para o
> experimento estar a minutos de lançar quando o veredito da v68 sair:
>
> - **Coletor de demos**: `scripts/collect_demos.py` roda episódios
>   sequenciais do professor scriptado no env Isaac gravando pares
>   (obs do grupo policy, ação) via `fanuc_cell.demos` (núcleo puro,
>   `.npz` com taxas de grasp/success por episódio). Limitação v1
>   honesta: coleta 1-env sequencial; professor vetorizado é futuro.
> - **BC**: `fanuc_cell.bc` — MLP numpy [256,128,64] ELU, espelho do
>   ator RSL-RL, Adam + MSE; `scripts/pretrain_bc.py` com portão de
>   qualidade das demos (success do professor < 0,80 → exit 3, sem
>   treino).
> - **Warm-start**: `scripts/train_ppo_bc.py` cria o runner padrão e
>   injeta os pesos por ordem de camadas (shapes conferidos, marcador
>   `BC_WARMSTART_APPLIED`); builder ganhou `--bc-warmstart` /
>   `--bc-episodes` (default desligado; kernel principal byte a byte
>   o mesmo sem a flag; slug separado `fanuc-bc-warmstart`).
> - **Protocolo pré-registrado na SPEC** (hipótese, gatilho =
>   veredito da v68, portões de demos e de fumaça, PROVA/PARCIAL/MATA
>   relativos a S68).
> - **Testes**: suíte 33/33 (7 novos: recorder/roundtrip, BC reduz a
>   perda em professor sintético, export casa com a arquitetura do
>   ator, portão rejeita demos fracas, fiação do builder).
>
> Falta para lançar: (1) veredito da v68 registrado (gatilho);
> (2) merge desta branch; (3) build com `--bc-warmstart` + push do
> kernel pelo caminho de sempre.
