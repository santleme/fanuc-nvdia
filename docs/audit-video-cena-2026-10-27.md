# Auditoria — Vídeo e cena do FANUC: do playback `newton_gl` ao vídeo industrial

Data: 27/10/2026 · Escopo: somente leitura sobre `~/workspace/fanuc-nvdia-git` (main `8bfaede`)
+ documentação oficial do Isaac Lab 3.0. Nenhum arquivo do repo foi alterado.

---

## 1. Diagnóstico da cena atual

### 1.1 A cena é 100% primitiva geométrica, sem materiais

Fonte única: `isaaclab_overlay/src/fanuc_cell_lab/tasks/conveyor_pick_place/config/m10id12/env_cfg.py`
(`CellSceneCfg`, linhas 68–177). Inventário:

| Entidade | Como é hoje | Asset/USD? |
|---|---|---|
| Chão | `GroundPlaneCfg(size=(4.0, 4.0))` — plano cinza default | não |
| Esteira (`belt`) | `CuboidCfg(size=(0.53, 0.84, 0.11))` — **uma caixa lisa** em (0.985, 0, 0.055) | não |
| Pedestal de deposição | `CuboidCfg(0.36, 0.32, 0.04)` | não |
| Peça (`part`) | `CuboidCfg(0.08³)`, massa 0,35 kg — cubo default | não |
| Robô | `UsdFileCfg` → USD convertido do URDF próprio (ver §1.2) | sim (primitivas) |
| Luz | 1× `DomeLightCfg(intensity=1800, color=0.8)` — dome neutro | — |
| Câmera | `VisualizerCfg(eye=(2.2, -2.5, 1.8), lookat=(0.8, -0.1, 0.3))` fixa (linha 290) | — |

Nenhum spawn declara `visual_material`: sem PBR, sem textura, sem roughness/metalness
em nada. O visual é o material default do visualizador sobre geometria nua — é por
isso que o vídeo atual parece "debug", mesmo sendo 1080p.

### 1.2 O robô: URDF de primitivas + cores chapadas, convertido no notebook

- `assets/m10id12_robot.urdf` (raiz do repo): todos os elos são **cilindros e caixas**
  (`<cylinder>`/`<box>`), com 4 materiais nomeados só por cor RGBA —
  `fanuc_yellow (0.96, 0.68, 0.08)`, `wrist_yellow`, `dark`, `metal`. Sem meshes,
  sem texturas, sem propriedades PBR (URDF nem expressa roughness/metalness).
  É o proxy declarado do projeto ("não é CAD oficial") — e visual = colisão
  (a mesma primitiva serve aos dois).
- A conversão URDF→USD **não é um script do repo**: acontece dentro do notebook
  Kaggle gerado por `scripts/build_kaggle_notebook.py` (linhas ~287–342): ele lê o
  `scripts/tools/convert_urdf.py` **oficial do Isaac Lab v3.0.0** (clonado na hora,
  `--branch v3.0.0`), aplica um patch de regex trocando o backend para `ovphysx`,
  converte com `--joint-target-type none` e depois aplica um **patch pxr** que liga
  `PhysxContactReportAPI` nos corpos rígidos do USD. O USD resultante herda as
  primitivas + cores do URDF (PreviewSurface difuso, nada além).
- `isaaclab_overlay/src/fanuc_cell_lab/assets/__init__.py` só expõe o caminho
  `assets/data/` onde o USD
...[truncated 13026 chars] convertido é gravado em runtime (não versionado).

### 1.3 O vídeo atual: pipeline e números

- Comando (builder, célula de playback): `isaaclab play --viz newton_gl --video
  --video_length 400 --num_envs 1 --seed 107 physics=ovphysx`, com
  `--extra ovphysx --extra video`. O `--video` popula um `VideoRecorderCfg` a
  partir do visualizador ativo; o mp4 cai em `logs/rsl_rl/**/videos/**` e é copiado
  para `/kaggle/working/fanuc-cell-isaaclab-policy.mp4`.
- Medido no artefato da v67 (`outputs-v67/fanuc-cell-isaaclab-policy.mp4`):
  **1920×1080, 50 fps, 400 frames, 8,0 s, 264 KB** (≈260 kbps). Resolução não é o
  problema: é o rasterizador OpenGL do Newton desenhando primitivas sem material,
  com uma dome light neutra e câmera fixa — compressível a quase nada porque a
  imagem é chapada.
- O teste `tests/test_isaaclab_scaffold.py` (linhas 136–137) fixa
  `'newton_gl', '--video` e `--video_length', '400` no fonte do builder — qualquer
  troca de visualizador precisa atualizar esse teste.

### 1.4 Separação visual × colisão: hoje não existe

Cada `CuboidCfg` gera **um** prim que é visual e colisão ao mesmo tempo. Não há
prim visual-only, mesh decorativa nem entidade de cenário fora das 4 caixas +
robô. Consequência prática: enriquecer a cena "no lugar" mexe nas entidades da
tarefa; o caminho seguro é uma camada visual ao redor (ver §4).

### 1.5 A "esteira" não se move — quem se move é a peça, por teleporte

`mdp/cell.py::advance_conveyor` (evento a cada 0,02 s) escreve a posição da peça
diretamente (transporte cinemático enquanto não agarrada, com a parada por
proximidade da correção pós-BC v3). A caixa-esteira é estática e nem é tocada
pelo MDP. Ou seja: "esteira de verdade" no vídeo = ilusão visual (correia/rolos
animados sincronizados à velocidade da peça), necessariamente desacoplada da
física — o que é legítimo e padrão, desde que confinada à camada de vídeo.

---

## 2. Opções reais de render no Isaac Lab 3.0.0 (fluxo kit-less)

O notebook instala o Isaac Lab **de fonte, kit-less** (clone `--branch v3.0.0` +
`uv`, extras `ovphysx`/`importers`/`video`). Nesse mundo, a documentação oficial
3.0 oferece três visualizadores relevantes (o visualizador é desacoplado da
física — qualquer um serve com o backend OvPhysX):

| Visualizador | Renderer | Precisa Isaac Sim? | Qualidade |
|---|---|---|---|
| `newton_gl` (atual) | Rasterizador OpenGL do Newton | não | Debug: flat, sem path tracing, materiais difusos |
| **`newton_rtx`** | **OVRTX — path tracer RTX** (pacote `isaaclab_ov`, extra `ovrtx`) | **não** | Sombras suaves, reflexos, GI por path tracing; renderiza offscreen |
| `kit` | Isaac Sim / Omniverse RTX completo | **sim** (Kit não está na imagem/fluxo kit-less) | Teto máximo (RTX real-time/path-traced, ecossistema USD completo) |

Fatos da documentação oficial (instalação 3.0 + release notes + API de envs):

- A imagem kit-less oficial 3.0 "carries Newton physics, OVPhysX physics,
  **OVRTX rendering**" e os visualizadores `newton_gl`, `newton_rtx`, `viser`,
  `rerun`; "`--viz newton_gl` selects the OpenGL rasterizer; `--viz newton_rtx`
  selects the OVRTX path tracer. Both can render offscreen without Isaac Sim."
- Tabela de renderers das release notes: OVRTX = "Kit-less RTX pipeline",
  requer `isaaclab_ov[ovrtx]`; Isaac RTX = requer Isaac Sim.
- Gravação: `video_recorders: list[VideoRecorderCfg]`; a fonte pode ser
  `"viz"` / `"viz:<tipo>"` (a câmera do visualizador) **ou sensores de câmera da
  cena** (`"sensor:<nome>:<canal>"` — rgb/depth/segmentation/normals). Com
  `--video`, a CLI cria automaticamente o recorder a partir do visualizador
  ativo → **trocar `--viz newton_gl` por `--viz newton_rtx` já faz o mesmo
  comando gravar o path tracer**, sem mudar a lógica do notebook.
- Pose/resolução da câmera vivem no cfg do visualizador
  (`VisualizerCfg(eye, lookat)` — já usado no `env_cfg`) ou em `CameraCfg` de
  sensores de cena, para ângulos cinematográficos independentes do viewport.

### Custos e riscos por opção (no Kaggle, T4 16 GB)

- **newton_rtx**: custo = tempo de render. Path tracing a 1080p com denoiser:
  estimativa de segundos/frame → os 400 frames viram **~5–40 min** de estágio
  (incerteza alta: depende de spp/denoiser e da cena; medir com um smoke de
  40 frames e extrapolar antes de comprometer o kernel). Dependência nova: o
  extra `ovrtx` do workspace — o **nome exato do extra e a classe de cfg do
  visualizador precisam ser confirmados contra a fonte pinada v3.0.0** no
  primeiro passo de implementação (a doc confirma o recurso e o pacote
  `isaaclab_ov[ovrtx]`; o fluxo de instalação aqui é uv a partir da fonte, não
  a imagem Docker). Risco ao pipeline: **zero se isolado no estágio de
  playback** (probe/baseline/treino/benchmark não usam visualizador —
  os scripts do repo rodam com `args.visualizer = "none"`).
- **kit**: exige instalar o Isaac Sim (pacotes de vários GB), aceitar EULA,
  compilar cache de shaders (dezenas de minutos na 1ª vez), e brigar com
  VRAM/tempo de kernel de 12 h. É o caminho "RTX de verdade" pelo qual o
  Isaac foi escolhido, mas **incompatível com o orçamento e a robustez do
  pipeline Kaggle atual**. Não recomendado agora.
- **Manter newton_gl**: custo zero, teto baixo — materiais melhores ajudam,
  mas sem sombras/reflexos de path tracing o resultado continua "simulador",
  não "vídeo industrial".

---

## 3. O que um upgrade de cena "industrial" envolve (sem quebrar a física)

Princípio: **a tarefa (treino/benchmark) não muda**. Todo o enriquecimento
mora numa variante usada só pelo estágio de vídeo:

1. **Variante cinematográfica da cena** — uma subclasse de `M10id12EnvCfg`
   (ou um segundo task id `…-Cinematic-v0`) que herda MDP/actions/rewards
   idênticos e sobrescreve só `scene`/`sim.default_visualizer_cfg`. O
   playback aponta para ela; treino, coleta e benchmark seguem na tarefa
   atual. Risco físico: nulo por construção.
2. **Esteira real (visual-only)**: manter o `CuboidCfg` atual como colisor
   (invisível na variante cinematic) e construir por cima, como entidades
   `AssetBaseCfg` **sem `collision_props`**: estrutura em perfil (2 trilhos
   laterais + pernas), rolos = cilindros ao longo do topo, correia = caixa
   fina escura com material PBR (roughness alto, leve textura procedural
   via `PreviewSurfaceCfg`). A física continua enxergando só a caixa de
   sempre — a separação visual×colisão passa a existir exatamente aqui.
3. **Animação da correia**: a peça desliza por teleporte a 0,05–0,10 m/s;
   a correia/rolos podem ficar estáticos na fatia 1 (o movimento da peça +
   do braço já vende a cena). Animação sincronizada (offset de UV da
   correia, rotação dos rolos) exige dirigir o env por script próprio —
   o repo já tem esse padrão (`scripts/run_scripted_demo.py`,
   `scripts/train_ppo_bc.py` dirigem o env fora do entrypoint) — fica
   para a fatia 2 como `scripts/render_cinematic.py` com checkpoint
   explícito, câmera animada e recorder `viz:newton_rtx`.
4. **Piso e set**: `GroundPlaneCfg` maior com `visual_material` concreto
   (cinza áspero, leve variação), faixa de segurança amarela (prim fina ou
   textura), cerca de segurança atrás da célula (postes + painel com
   material semi-transparente quadriculado), um pedestal/mesa mais
   trabalhado para a zona de deposição. Tudo visual-only, dentro do
   namespace do env (o playback roda com `num_envs=1`, sem custo de clone).
5. **Iluminação de galpão**: manter a `DomeLightCfg` como ambiente e somar
   luz-chave (`DistantLightCfg`/`RectLightCfg` via `sim_utils`) para
   sombras direcionais no path tracer; fundo mais escuro que o atual para
   o amarelo FANUC destacar.
6. **Materiais PBR nos primitivos da cena**: `CuboidCfg`/primitivas aceitam
   `visual_material=PreviewSurfaceCfg(diffuse_color, roughness, metallic)`
   — correia quase-preta áspera, estrutura metálica (metallic alto,
   roughness médio), peça com cor/metaleidade distintas do ambiente.
7. **O robô**: o amarelo vem do URDF e o conversor não gera PBR. Opções:
   (a) fatia 1: deixar como está — sob path tracing + iluminação nova, o
   difuso FANUC já ganha sombra/reflexo críveis; (b) fatia 2: pós-processar
   o USD convertido no **mesmo ponto onde o notebook já faz o patch pxr**
   (patch de contato), atribuindo roughness/metalness por nome de material
   (`fanuc_yellow` → pintura: roughness ~0,35; `metal` → metallic 1,0).
   Trocar as primitivas por meshes reais do M-10iD fica **fora** de
   recomendação: conflita com o contrato declarado do projeto (proxy, não
   CAD) e exigiria overlay visual alinhado elo a elo mantendo a colisão
   atual — trabalho grande, ganho incerto.
8. **Câmera**: fatia 1 = uma pose fixa bem escolhida (mais baixa e perto do
   pick que o atual (2.2, -2.5, 1.8), enquadrando esteira→pick→pedestal).
   Trajetória (dolly/orbit) e cortes multi-ângulo: via recorder com fonte
   em sensores `CameraCfg` ou script da fatia 2.

---

## 4. Rotas de upgrade

### Rota 1 — Só cena e materiais, mantendo `newton_gl`
- **O quê**: variante cinematic (§3 itens 2, 4, 5, 6, 8) sem trocar o renderer.
- **Esforço**: baixo-médio (1 sessão: env_cfg variant + builder apontando o
  playback). **Risco**: mínimo. **Custo de render**: igual ao atual (segundos).
- **Teto**: moderado. O rasterizador GL não faz path tracing: sem reflexos/GI,
  o resultado melhora de "debug" para "simulador apresentável" — **não** chega
  a "vídeo industrial insano".

### Rota 2 — `newton_rtx` (OVRTX) + cena cinematic — **RECOMENDADA**
- **O quê**: Rota 1 + troca do visualizador do playback para `newton_rtx`
  (extra `ovrtx` no comando) + smoke de render curto antes do playback cheio.
- **Esforço**: médio (a rota 1 + 1 flag/extra + calibração de tempo de render).
  **Risco**: médio-baixo e contido — falha possível no extra/classe de cfg da
  fonte pinada (detectável no smoke, antes de gastar o kernel) e tempo de
  render maior (mitigado medindo 40 frames e ajustando resolução/spp).
  **Custo de render**: ~5–40 min estimados para 400 frames 1080p no T4.
- **Teto**: alto para o fluxo kit-less — é o pipeline RTX oficial sem Isaac
  Sim: sombras suaves, reflexo no piso metálico, leitura "industrial" real.

### Rota 3 — Isaac Sim completo (`--viz kit`) no Kaggle
- **O quê**: instalar o ecossistema Kit no notebook e renderizar com RTX do
  Isaac Sim.
- **Esforço**: alto. **Risco**: alto (tamanho de install, EULA, shader cache,
  VRAM, tempo de kernel; briga com o desenho kit-less que sustenta todo o
  pipeline atual). **Custo de render**: comparável ao OVRTX por frame, mas
  com overhead de boot muito maior.
- **Teto**: o maior em teoria — mas desproporcional agora; reconsiderar só
  se um dia o vídeo virar produto fora do Kaggle (workstation/VM com GPU
  RTX de verdade).

*(Horizonte, fora das rotas: replay offline — gravar estados da articulação/
peça num playback barato e re-renderizar quadro a quadro com câmera
cinematográfica livre, inclusive fora do Isaac. É o formato "filme" completo,
mas é outro projeto; a Rota 2 entrega 80% do impacto com 20% desse custo.)*

---

## 5. Recomendação e escopo da "fatia 1"

**Rota 2, em fatia mínima executável:**

1. **Builder** (`scripts/build_kaggle_notebook.py`): flag `--cinematic-video`
   (default desligada) que, só na célula de playback, troca
   `'--viz', 'newton_gl'` → `'--viz', 'newton_rtx'` e adiciona o extra do
   OVRTX ao comando `uv run`. Atualizar o teste do scaffold
   (`tests/test_isaaclab_scaffold.py:136-137`) para o novo default.
2. **Pré-voo do OVRTX** (primeiro passo de implementação, antes de qualquer
   kernel): na fonte pinada v3.0.0, confirmar nome do extra, classe de cfg
   do visualizador `newton_rtx` e que `isaaclab play --viz newton_rtx
   --video` grava offscreen; smoke de 40 frames medindo s/frame no T4 e
   extrapolando os 400 (se estourar o orçamento, baixar p/ 720p no recorder
   ou reduzir spp — decisão com número, não no escuro).
3. **Variante cinematic** em `env_cfg.py`: subclasse da env cfg atual com
   (a) `visual_material` PBR em esteira/pedestal/peça/piso; (b) set
   visual-only mínimo: pernas + trilhos + rolos estáticos da esteira, piso
   maior com faixa de segurança, cerca simples ao fundo; (c) luz-chave
   somada à dome; (d) câmera fixa nova enquadrando pick→place. A tarefa de
   treino/benchmark **não é tocada**; física idêntica por herança.
4. **Aceite da fatia 1**: kernel com a flag gera o mp4 em `newton_rtx` sem
   alterar probe/baseline/treino/benchmark; vídeo visualmente distinto do
   atual (sombras, reflexo, esteira estruturada); tempo total do kernel
   dentro do teto de 12 h com folga medida pelo smoke.
5. **Fora da fatia 1** (fatia 2, se o fundador aprovar o salto): script
   `render_cinematic.py` com correia/rolos animados sincronizados à
   velocidade da peça, trajetória de câmera, patch PBR no USD do robô e
   cortes multi-ângulo via `CameraCfg`.

---

## Fontes

- Repo (leitura direta): `isaaclab_overlay/.../config/m10id12/env_cfg.py`,
  `isaaclab_overlay/.../mdp/cell.py`, `assets/m10id12_robot.urdf`,
  `assets/cell_geometry.json`, `scripts/build_kaggle_notebook.py`,
  `src/fanuc_cell/render.py`, `scripts/render_demo.py`,
  `tests/test_isaaclab_scaffold.py`, artefato `outputs-v67/` (ffprobe).
- Documentação oficial Isaac Lab 3.0 (busca web, 27/10/2026): página de
  instalação (imagem kit-less, OVRTX, `--viz newton_gl` × `--viz newton_rtx`),
  release notes 3.0 (tabela de renderers: Isaac RTX × OVRTX × Newton Warp;
  sistema de visualizadores plugáveis), API `isaaclab.envs`
  (`video_recorders`/`VideoRecorderCfg`, fontes `viz:<tipo>` e
  `sensor:<nome>:<canal>`) e API `isaaclab.sim` (`visualizer_cfgs`,
  `default_visualizer_cfg`).
- Ressalva de precisão: tempos de render OVRTX no T4 são **estimativas**
  (não medidos); nome exato do extra uv e da classe de cfg do `newton_rtx`
  na fonte pinada v3.0.0 devem ser confirmados no pré-voo (item 2 da fatia 1).
