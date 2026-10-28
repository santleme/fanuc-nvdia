# Cache da instalação do Isaac Lab em dataset Kaggle (issue #12, B4)

## Problema

Cada kernel do projeto paga o clone do Isaac Lab v3.0.0 + `uv sync`
completo do zero: **~4–6 min por corrida** (medido no veredito v67).
Com a cadência atual de experimentos, o payback do cache é de ~6
corridas (estimativa registrada na SPEC, protocolo v67).

## Desenho

Um **kernel construtor** paga a instalação uma vez e a publica como
dataset; os kernels consumidores **montam o dataset e restauram** a
árvore instalada em vez de instalar. Qualquer falha na restauração
cai na instalação completa atual — o cache é uma otimização, nunca
um novo ponto de falha.

```
construtor (fanuc-isaaclab-cache-builder, 1× por versão do Isaac Lab)
  clone v3.0.0 → uv sync (UV_LINK_MODE=copy) → aquece extras
  → scripts/prepare_isaac_cache.py fotografa no layout do dataset
  → saída do kernel vira o dataset `fanuc-isaaclab-cache`

consumidor (build_kaggle_notebook.py --isaac-cache-dataset USUARIO/fanuc-isaaclab-cache)
  metadata ganha dataset_sources → Kaggle monta em /kaggle/input/fanuc-isaaclab-cache
  célula de instalação: manifesto válido? → copia árvore + cache do uv
    → sonda `uv run --no-sync python -c "import isaaclab, torch"`
    → sonda ok: segue o kernel; sonda/montagem falhou: clone+sync de sempre
```

A decisão de restaurar é do núcleo puro `src/fanuc_cell/isaac_cache.py`
(manifesto, descoberta do mount, plano de cópia como dados) — o
notebook só executa. A sonda de import é o portão final: manifesto
válido + cópia ok + sonda ok = cache; qualquer outra combinação =
instalação completa, registrada em `runtime_report['isaac_cache']`
(`mode`: `restored` / `absent` / `probe_failed` / `error`) com o
tempo em `timings_seconds['isaac_setup']` — a medição da aceitação
da issue (setup com cache × durações de clone+sync da linha de base).

## Layout do dataset

| caminho | conteúdo |
|---|---|
| `isaac-cache-manifest.json` | manifesto: formato (v1), repo/ref do Isaac Lab, python/plataforma do construtor, extras aquecidos, partes com bytes |
| `isaaclab/` | checkout v3.0.0 **com a `.venv`** do `uv sync` (sem `.git` nem `logs`) |
| `uv-cache/` | o `~/.cache/uv` do construtor, para os `uv run --extra ...` resolverem sem baixar nada |

Decisões que sustentam o layout:

- **`UV_LINK_MODE=copy` no sync do construtor**: a `.venv` fica
  autossuficiente. Com o modo default (hardlink para o cache do uv),
  a árvore restaurada de um mount somente-leitura quebraria.
- **Restaurar para o MESMO caminho absoluto** (`/kaggle/working/IsaacLab`,
  `~/.cache/uv`): scripts e `pyvenv.cfg` da venv carregam caminhos
  absolutos; construtor e consumidores compartilham a imagem do
  Kaggle, então o caminho de construção é o caminho de restauração.
- **Extras aquecidos** (`ovphysx`, `video`, `importers`, `ovrtx`):
  são os que os kernels do projeto resolvem via `uv run --extra`;
  aquecê-los no construtor significa que nem a primeira corrida
  restaurada baixa pacotes.
- **Manifesto amarra a versão**: formato ≠ 1 ou `isaaclab_ref` ≠
  `v3.0.0` ⇒ cache recusado e instalação completa. Bump do Isaac Lab
  = rodar o construtor de novo e publicar versão nova do dataset —
  nunca adaptar um cache velho.
- **Override de mount**: `FANUC_ISAAC_CACHE_DIR` no ambiente do
  kernel aponta para um mount alternativo (diagnóstico); o default
  é o mount do dataset declarado no metadata.

## Como ativar (passos de corrida, fora do código)

1. Gerar e subir o kernel construtor:
   `python scripts/build_cache_notebook.py --kaggle-user pascalcoimbra`
   e push do kernel `fanuc-isaaclab-cache-builder` (o script de push
   lê `kaggle/`; para este kernel secundário vale o mesmo arranjo
   manual de metadata usado pelo kernel de diagnóstico).
2. Rodar o construtor no Kaggle (CPU basta; precisa de internet).
   A saída `isaaclab-cache/` traz o manifesto verificado.
3. Criar o dataset privado `pascalcoimbra/fanuc-isaaclab-cache` a
   partir da saída do kernel (UI do Kaggle: *Create Dataset* a
   partir do output, ou `kaggle datasets create` sobre a pasta
   baixada).
4. Gerar os kernels consumidores com o cache:
   `python scripts/build_kaggle_notebook.py --kaggle-user pascalcoimbra --isaac-cache-dataset pascalcoimbra/fanuc-isaaclab-cache ...`
5. Aceitação da issue: primeira corrida completa com
   `runtime_report['isaac_cache']['mode'] == 'restored'`, comparando
   `timings_seconds['isaac_setup']` com as durações clone+sync das
   corridas sem cache.

## Limites conhecidos

- A venv é da imagem do Kaggle: se o Kaggle trocar a imagem
  (python/driver), a sonda de import falha e TODOS os kernels caem
  para a instalação completa até o construtor rodar de novo — o
  fallback torna a troca de imagem um não-evento.
- O dataset tem vários GB (venv com torch/ CUDA + cache do uv);
  reconstruir a cada bump do Isaac Lab é o custo esperado.
- Kernels **sem** `--isaac-cache-dataset` geram células e metadata
  idênticos aos de sempre: o cache é opt-in por build, não uma
  mudança de default (só o zip do bundle embutido ganha os arquivos
  inertes do núcleo `isaac_cache` e do script de preparo).
