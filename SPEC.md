# Spec: célula FANUC de pick-and-place em esteira

**Status:** aprovado para implementação inicial  
**Revisão:** 1  
**Data:** 2026-10-25

## Objetivo do produto

Construir uma célula industrial virtual programável com um braço articulado de
seis eixos inspirado no FANUC M-10iD/12. Uma peça se desloca numa esteira, o
robô a intercepta com uma garra e a deposita numa zona definida. O projeto deve
rodar em notebook Kaggle e produzir um vídeo verificável.

Isaac Lab 3.0 com OvPhysX kit-less é o alvo de simulação e treinamento. O primeiro
comportamento de referência será determinístico e programado; aprendizado por
reforço será uma evolução medida contra essa referência, não um pré-requisito
para a célula conseguir pegar uma peça.

## Usuários e uso previsto

- O autor desenvolve cinemática, controle, lógica da esteira e, posteriormente,
  políticas de robótica.
- O Kaggle fornece GPU NVIDIA remota e executa um notebook reprodutível.
- Os clipes mostram o comportamento do modelo virtual e não certificam ciclo,
  alcance, colisões ou segurança de uma célula física.

## Base de referência e limites de fidelidade

O robô de referência escolhido para iniciar é o FANUC M-10iD/12: 6 eixos,
12 kg de capacidade nominal no punho, alcance de 1.441 mm e repetibilidade
publicada de ±0,02 mm. Fonte primária:
[FANUC America](https://www.fanucamerica.com/products/robot/m-10id-12).

O primeiro modelo terá dimensões de elos e inércias aproximadas, configuradas
explicitamente e rotuladas como aproximação. A página de produto não fornece
geometria articulada suficiente para reconstruir com precisão o robô. Para
fidelidade de produção serão necessários modelo CAD/URDF oficial, ferramenta,
payload, controlador e configuração de montagem da célula.

## Ambiente e arquitetura

- Isaac Lab **v3.0.0** fixado como baseline oficial recém-lançado.
- Execução kit-less com Isaac Lab OvPhysX; Isaac Sim completo
  fica fora do caminho padrão do Kaggle.
- Python 3.12, ambiente `uv`, CUDA NVIDIA.
- O T4 tem compute capability 7.5 e satisfaz o requisito arquitetural mínimo
  Turing do stack CUDA. O driver presente no Kaggle ainda precisa ser
  verificado em runtime; não presumir que a GPU basta para satisfazer todos os
  requisitos da instalação.
- T4x2 significa dois dispositivos de 16 GB, sem memória unificada. A primeira
  execução usará uma GPU; paralelismo entre GPUs só será adotado se profiling
  mostrar benefício.
- TPU não é backend CUDA para Isaac Lab e não participa deste fluxo.

Fontes: [release Isaac Lab 3.0.0](https://github.com/isaac-sim/IsaacLab/releases),
[instalação/requisitos Isaac Lab](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/isaaclab/installation.html),
[capacidade CUDA T4](https://developer.nvidia.com/cuda/gpus),
[hardware Kaggle T4x2](https://www.kaggle.com/product-announcements/735239).

## Requisitos funcionais

1. **Célula:** incluir base, manipulador serial de seis eixos, garra, esteira,
   uma peça e zona de depósito identificáveis.
2. **Movimento:** comandar posições articulares limitadas e mover o efetuador
   entre aproximação, captura, elevação, transferência e soltura.
3. **Esteira:** avançar a peça com velocidade configurável; permitir uma política
   de intercepção baseada na posição medida da peça e velocidade da correia.
4. **Captura:** permitir fechar/abrir a garra e verificar que a peça está presa,
   transportada e liberada na zona final.
5. **Treinamento:** fornecer tarefa Isaac Lab compatível com o backend escolhido,
   com observações, ações, recompensas, reset e critério de sucesso explícitos.
   A política treinada deve ser comparada com o controlador determinístico.
6. **Vídeo:** salvar um clipe headless no notebook; armazenar o arquivo como
   output do kernel e incluir overlay ou log com estado da tarefa.
7. **Reprodutibilidade:** registrar versão do Isaac Lab, GPU, driver, seed,
   parâmetros físicos, número de ambientes, comando e resultado de execução.
8. **Kaggle:** disponibilizar o projeto num notebook/kernel privado via CLI,
   quando a conectividade da API estiver disponível; não incluir credenciais em
   arquivos ou outputs.

## Requisitos não funcionais

- A inicialização do notebook deve detectar falta de GPU/driver e falhar com
  diagnóstico acionável, sem erro opaco de importação.
- A simulação padrão deve manter um ambiente para depuração; treinamento poderá
  aumentar ambientes respeitando VRAM e quota do Kaggle.
- O código da tarefa deve ser independente de metadados secretos Kaggle.
- Unidades internas SI: metros, segundos, quilogramas, radianos.
- O clipe deve ser gerado por renderização suportada no modo headless; não exigir
  uma janela gráfica no notebook.

## Critérios de aceite da célula completa

- [x] Uma execução limpa do notebook detecta e relata as GPUs e o driver
      realmente disponibilizados pela sessão Kaggle.
- [x] Instalação fixada de Isaac Lab 3.0.0 e inicialização de OvPhysX
      completam no runtime Kaggle selecionado.
- [ ] O asset importado contém seis juntas revolutas na ordem esperada e limites
      aproximados declarados no proxy do modelo.
- [ ] O robô alcança pontos de aproximação, pick e place sem violar limites e
      sem cruzar a geometria estática da célula.
- [ ] A peça se move com a esteira, é agarrada, transportada e liberada na zona
      de destino; o episódio informa sucesso/falha de forma explícita.
- [ ] O notebook salva um vídeo legível e um resumo JSON dos resultados.
- [ ] Pelo menos um clipe de execução real do Kaggle é baixado e inspecionado;
      nenhum sucesso remoto será declarado com base só em validação local.
- [ ] O comportamento determinístico passa seus testes antes de treinar uma
      política; a política é avaliada em seeds e offsets de peça não vistos.
- [ ] O notebook privado é enviado e o status da execução é terminal/sucesso.

## Progresso verificado

- **Feito localmente:** spec inicial; proxy serial configurável de 6 eixos com
  limites aproximados; IK posicional e de pose completa com orientação
  nivelada; offset TCP-peça explícito; estados e waypoints de interceptação e
      soltura; URDF com seis juntas da arm, dois slides de garra com abertura para
  peça de 8 cm, esteira e peça; task package externo registrado no Gymnasium,
  manager-based, com agentes PPO e sensores filtrados de contato nos dedos; o
  grasp exige contato bilateral, fechamento, levantamento e soltura na zona. O
  notebook privado está configurado para PPO, avaliação e vídeo headless.
- **Evidência local:** `PYTHONPATH=src pytest -q` passa 21 testes. A execução
  `scripts/render_demo.py --output outputs/fanuc-cell-demo.mp4 --fps 8` gera um
  MP4 de 45 frames, lido de volta pelo OpenCV; o JSON final marca sucesso e erro
  de interceptação ~0,06 mm. Os testes incluem contratos estáticos do registro
  da tarefa/sensores/metadata e retorno dos termos de sucesso/falha. Notebook passa
  validação nbformat e parsing AST das células e confirma que o pacote contém o
  controlador físico, a métrica de sucesso e as configs registradas. Um teste
  geométrico amostra elos, punho, ferramenta e dedos contra esteira e pedestal;
  a folga do proxy fica acima de 5 mm. Essa checagem não substitui o contato no
  solver OvPhysX.
- **Evidência Kaggle:** as versões 1–8 foram enviadas pelo CLI para
  `pascalcoimbra/fanuc-conveyor-pick-place`. O runtime informou 2× Tesla T4
  (15.360 MiB cada), driver 580.178.04 e Python 3.13.15. Isaac Lab 3.0.0,
  OvPhysX e o smoke Cartpole passaram. A importação URDF produziu o asset e a
  execução validou juntas e corpos antes de inicializar a tarefa customizada.
- **Falhas remotas corrigidas no código local:** a conversão URDF não ativa
  contact reporting por configuração; o notebook agora aplica as APIs PhysX
  nos rigid bodies USD. O reset/evento também aceita `env_ids=slice(None)`, e
  `scene.env_origins` é tratado como tensor, conforme a API 3.0.0.
- **Execuções Kaggle:** a versão 8 revelou e permitiu corrigir o reset
  (`env_ids=slice(None)` e `scene.env_origins` tensor). A versão 12 confirmou
  smoke Isaac Lab/OvPhysX e da tarefa, mas falhou no baseline por uma flag
  `--headless` não aceita pelo launcher. A versão 14 passou esses smokes, mas
  falhou ao construir o baseline por receber `CellPhysicsCfg` em vez da
  configuração concreta de física resolvida pelo launcher. A versão 18 passou
  o smoke Cartpole e criou o ambiente da célula; o baseline físico executou
  todas as fases, mas a pinça não registrou contato bilateral (forças 0 N), o
  objeto não foi agarrado e o baseline retornou falha. A versão 19 está salva
  sem execução e registra juntas reais e TCP por fase para diagnosticar o erro.
  A versão 20 revelou que os alvos de junta eram limitados a ±1 rad após a
  escala; a versão 22 corrigiu os limites, mas ainda não agarrou a peça. A
  versão 24 aumentou o tempo da sequência e confirmou que o servo tinha erro
  estático sob gravidade. A versão 26 elevou rigidez/esforço e tornou o avanço
  da esteira determinístico; a 27 alterou o offset do TCP e a rigidez. A
  versão 28 passou o preflight em 2× Tesla T4, os testes do pacote local, o MP4
  cinemático, o smoke Cartpole oficial e o smoke da tarefa customizada. O
  baseline físico ainda falhou no grasp, sem força de contato. Na 28, a peça
  estava em x=1,104 m aos 3 s (interceptação calculada: 1,039 m); zerar a
  velocidade longitudinal depois da atualização de pose retirou a
  supervelocidade, mas a revisão 29 revelou avanço insuficiente: x=0,991 m aos
  3 s. O movimento via eventos de intervalo não correspondeu ao relógio do
  episódio. A 29 também mediu deflexão de ~7 cm no TCP durante a captura.
  A revisão 30 confirmou a previsão por contador de passos: a peça atingiu
  x=1,040 m aos 3 s. O baseline mediu forças de contato máximas de 425 N e
  375 N nos sensores dos dedos, mas não levantou a peça; aos 4,7 s ela ainda
  estava em z=0,152 m. A análise mostrou que o critério exigia abertura de dedo
  >0,018 m enquanto a pose indicava cerca de 0,013 m, e o evento continuava
  escrevendo a pose da peça sobre a esteira enquanto havia contato. A versão
  33 foi salva sem execução pelo Kaggle CLI: baixa o limiar para 0,009 m, pausa
  o avanço cinemático quando há pinça bilateral e acompanha a peça até a posição
  de início da elevação, mantendo esse alvo fixo durante o lift. Os testes locais
  passam (21). `kaggle kernels pull .../33` recuperou o notebook salvo; a
  comparação das sete células confirmou igualdade com a fonte local. Ainda
  falta executar a versão salva: o Kaggle reporta as sessões
  29 e 30 como `RUNNING` mesmo com logs Papermill terminais de falha, e `kernels
  push` respondeu `Maximum batch GPU session count of 2 reached`. O CLI instalado
  não tem comando de cancelamento. Testei `CancelKernelSession` pelo SDK oficial
  distribuído com a CLI; o `ACCESS_TOKEN` atual recebeu HTTP 403 (`Permission
  'kernelSessions.cancel' was denied`). O login OAuth da CLI solicita
  `resources.admin:*`, mas a autenticação requer navegador e entrada
  interativa; a tentativa sem TTY terminou em EOF antes de alterar credenciais.
  Após as sessões 29/30 encerrarem, a versão 34 foi enviada e iniciou execução
  GPU pela CLI; o preflight detectou 2× Tesla T4 e os testes/vídeo cinemático
  passaram. A 34 terminou com erro no baseline: forças máximas de 403 N e
  330 N, mas nenhum grasp/lift; a peça avançou de acordo com a esteira até
  x=1,158 m durante a fase de captura. Os snapshots indicam que os sensores
  podem não ter registrado contato simultâneo antes de a peça escapar. A revisão
  35 registrou contato bilateral e fechado no snapshot de hold (108/70 N), mas
  a peça continuou indo à frente quando a distância TCP-peça oscilou e a esteira
  retomou o controle. A versão 36 mantém a pausa de captura latched até o reset
  do episódio. A versão 36 terminou em falha: 77 passos com contato bilateral
  fechado, mas a peça não subiu. A análise do URDF encontrou outro provável
  erro: o script pedia 25 mm por dedo, embora a peça de 80 mm encoste por volta
  de 6 mm. A versão 37 foi salva sem execução pelo Kaggle CLI com fechamento
  parcial, limiar de captura compatível e diagnóstico da pausa da esteira. O
  push com execução segue bloqueado pelo limite de duas sessões; as versões
  35/36 continuam listadas como RUNNING apesar dos logs terminais. O CLI não
  oferece cancelamento e a chamada SDK retornou HTTP 403. PPO segue condicionado
  ao sucesso físico. A v42 mediu 81,9 s no `uv sync` do Isaac Lab, 50,5 s no
  smoke Cartpole, 57,2 s na sincronização do projeto customizado e 21,1 s no
  setup local/render. Um pedido explícito de L4 ainda recebeu 2×T4. A versão 43
  está salva sem execução com T4 padrão: pula o render cinemático redundante,
  remove instalação de extras já presentes e roda o baseline antes de instalar
  o ambiente PPO ou repetir smokes. O alvo da garra subiu a 14 mm após 10 mm
  gerar apenas ~2 N por dedo. O push de execução da v43 foi recusado porque o
  Kaggle ainda conta v40/v42 como sessões `RUNNING` apesar dos logs terminais.
  O vídeo local é cinemático, sem
  contato ou dinâmica, e não prova que a garra física consegue carregar a peça.

### Estado atual — v46 e otimização da depuração

A v44 rodou com o alvo da garra em 14 mm e pausou a esteira, mas a posição TCP
continuou seguindo a peça deslocada pelo contato. A v46 fixou a posição da peça
no instante da captura e removeu esse arrasto: a âncora ficou em x=1,043 m e a
peça terminou em x=0,967 m, sem acompanhar a esteira durante a elevação. O
baseline ainda falhou: houve contato bilateral fechado por somente 4 passos,
com picos de 231 N e 415 N; no snapshot de hold um dedo estava em 14 mm e o
outro praticamente em zero. A análise do script revelou que uma compensação
manual de J2/J3 alterava o TCP em cerca de 4 cm depois da IK. Também, os dedos
de 18 mm de espessura com alvo de 14 mm avançavam além do necessário para a
peça de 80 mm.

A fonte local preparada para a próxima revisão remove a compensação de J2/J3 e
comanda 6,25 mm por dedo, deixando 80 mm entre as faces internas dos dedos.
O notebook foi regenerado, mas o `kaggle kernels push -p kaggle` foi recusado
pela cota `Maximum batch GPU session count of 2 reached`; status e logs da CLI
continuam mostrando as execuções anteriores como `RUNNING` apesar do erro
terminal da Papermill. Não foi possível enviar esta revisão ainda. A v46 mediu
86,5 s para preparar Isaac Lab, 16,8 s para converter URDF e 53,4 s no baseline
físico que falhou. A otimização já removeu do caminho de depuração o smoke
Cartpole, o smoke customizado e o render cinemático; o treino PPO só sincroniza
suas dependências depois de o baseline passar.

A v48 executou essa correção no T4: 80 passos com contato bilateral, mas só
~0,12 N por dedo. A v49 aumentou o curso para 8 mm e elevou o contato para
~1 N por dedo, ainda insuficiente para sustentar a peça de 350 g. A v50 foi
salva sem execução na CLI com curso-alvo de 12 mm e rigidez do drive da garra
de 2000 N/m (limite de esforço mantido em 40 N). A execução da v50 está
pendente: a v49 aparece `COMPLETE`, porém o Kaggle CLI ainda responde
`Maximum batch GPU session count of 2 reached`, indicando sessões GPU fora da
execução concluída deste kernel.

A v51 (mesmo código da v50) confirmou grasp físico, elevação até z=0,303 m e
depósito no pedestal em [0,894, -0,517, 0,280] m. Houve contato bilateral
fechado em 229 passos. A regra de sucesso anterior exigia distância radial de
7 cm ao centro; o depósito ficou a 11,3 cm, mas dentro da plataforma e com o
objeto apoiado. A v52 definiu uma zona segura de raio 12 cm, baseada no
pedestal 36×32 cm e peça 8 cm, mas o termo de sucesso encerrou o ambiente e o
autoreset apagou o estado antes da coleta do relatório. A v53 desliga somente
os resets automáticos de sucesso/perda no rollout de referência; ele calcula
sucesso no estado físico final. Esses termos seguem ativos no ambiente de PPO.
A v53 foi salva pela CLI com `--no-run`; a v52 está `COMPLETE`, mas a CLI
continua recusando novo job com `Maximum batch GPU session count of 2 reached`.
Ainda falta executar a v53, validar o baseline e então permitir treino PPO e
vídeo Isaac Lab.

A v54 executou a v53 e o baseline físico passou. O PPO falhou imediatamente
com `NaN` no grupo de observação `policy`. A v55 reduziu o desvio padrão de
exploração de 0,7 para 0,1 rad, mas o NaN persistiu já no primeiro passo. O
stdout mostrou avisos ao clonar a articulação raiz em 32 ambientes. Para
isolar esse comportamento, a v56 reduz o rollout PPO para 8 ambientes e mantém
o log do treino no relatório quando falha. A v56 foi salva sem execução; a CLI
voltou a responder `Maximum batch GPU session count of 2 reached`, apesar da
v55 aparecer `COMPLETE`. Próximo passo: executar a v56 quando a cota batch
liberar e conferir se o NaN dependia do clone de 32 ambientes.

A v57 confirmou que o NaN também ocorre com 8 ambientes. A v58 falhou antes
do baseline porque o script novo de diagnóstico não estava na lista de arquivos
do bundle; esse empacotamento foi corrigido em `build_kaggle_notebook.py`. A
v59 foi salva sem execução com o probe incluído: ele checa 24 passos de ações
pequenas e aponta índices/campos não finitos antes de iniciar PPO. A execução
da v59 ainda está pendente: apesar da v58 constar `COMPLETE`, a CLI responde
`Maximum batch GPU session count of 2 reached`.

A v60 corrigiu o bundle e executou o probe de 8 ambientes: as observações ficaram
finitas, mas o vetor de reward já continha `NaN` no primeiro passo. A v61 amplia
o probe para imprimir os valores individuais do RewardManager nos ambientes
afetados. Ela foi salva sem execução; novas tentativas pela CLI continuam
recusadas pelo limite `Maximum batch GPU session count of 2 reached`, mesmo
com a v60 `COMPLETE`.

A v62 identificou `reach` como a parcela inválida nos ambientes 1–7. A v63
registrou a causa: `tool0` e `part_relative_to_tcp` têm coordenadas `NaN` nesses
clones; o ambiente 0 está finito. A origem é `replicate_physics=True` clonando
o `root_joint` fixo sem `body0` rel sob OvPhysX. A v64 desativa replicação de
física para instanciar cada cena separadamente e manter o PPO com 8 ambientes.
Ela foi salva sem execução. O Kaggle CLI ainda responde
`Maximum batch GPU session count of 2 reached`, embora a v63 esteja
`COMPLETE`; falta executar a v64 para verificar o probe e seguir ao PPO.

## Plano SDD e evidência

### Fase 0 — especificação e viabilidade do runtime

Confirmar a oferta Kaggle, CLI, cota, GPU, driver, arquitetura CUDA, Python e
tempo de sessão. Criar kernel mínimo que detecta os dispositivos e salva o
relatório. Evidência: log Kaggle e metadados do runtime.

### Fase 1 — modelo e comportamento de referência

Criar o asset 6-DOF aproximado, a geometria de esteira/peça/garra e o movimento
determinístico. Testar limites, cinemática, interceptação e estados da tarefa.
Gerar o primeiro clipe curto. Evidência: testes, log e MP4.

### Fase 2 — tarefa Isaac Lab

Empacotar a tarefa externa Isaac Lab, instalar e executar com OvPhysX em
um ambiente. Validar reset, contatos, garra e critérios de sucesso. Evidência:
smoke run remoto e métricas por episódio.

### Fase 3 — treinamento

Adicionar PPO/RSL-RL, variar a posição inicial e velocidade da esteira e medir
taxa de sucesso, ciclos inválidos e tempo de treino. Comparar política e baseline.

### Fase 4 — entrega no Kaggle

Enviar notebook privado usando Kaggle CLI, acompanhar logs/status, recuperar os
outputs, inspecionar o vídeo e guardar os comandos e resultados no projeto.

## Decisões e riscos

- **D1:** Isaac Lab 3.0.0 + OvPhysX para executar PhysX sem inicializar Omniverse
  Kit. A sessão Kaggle é a autoridade para o modelo de GPU e memória realmente
  alocados.
- **D2:** controle determinístico antes do RL para que o treino tenha baseline
  verificável e para não exigir aprendizagem onde cinemática resolve o problema.
- **D3:** aproximar a geometria no início, sem reivindicar equivalência ao FANUC
  físico. Trocar para CAD/URDF autorizado antes de sim-to-real.
- **D4:** tratar a célula como projeto externo Isaac Lab com workflow
  manager-based single-agent e RSL-RL/PPO. Primeiro executar o baseline
  determinístico sob física e validar os contatos; só então treinar e comparar
  a política, conforme o template oficial da versão 3.0.0.
- **R1:** o driver/cuda ou a versão da imagem do Kaggle pode não satisfazer o
  ambiente Isaac Lab 3.0 mesmo com T4 compatível por compute capability.
- **R2:** a renderização Newton GL e a gravação headless precisam ser validadas
  na imagem Kaggle; esse componente só visualiza a cena e não é o solver físico.
- **R3:** execuções Kaggle com erro podem demorar a atualizar o estado terminal;
  a CLI é a interface para envio, status, logs e recuperação de outputs.

### Estado atual — diagnóstico de clone modes e v66 (26/10/2026)

A v65 executou a correção da v64 pela primeira vez e falhou na criação do
ambiente: com `replicate_physics=False`, o binding nativo de contato do
OvPhysX omitiu os corpos `left_finger` dos envs 1–7 (só o env_0 vinculou) e
baseline/PPO nunca rodaram. Para não trocar um chute por outro, o protocolo
pré-registrado do kernel de diagnóstico `pascalcoimbra/fanuc-diagnose-clone-modes`
v1 rodou cinco modos em processos separados (8 envs, 24 passos, seed 7;
script `scripts/diagnose_clone_modes.py`, veredito impresso pelo próprio
kernel):

- **A** (replicate=True, filtrado, USD com `--fix-base`): criou e andou, mas
  TODOS os corpos (base, elos, tool0, dedos) ficaram NaN nos envs 1–7 após os
  passos; reward não-finito. O NaN não era só do tool0.
- **B** (= v65): falhou na criação, binding de contato omitido nos envs 1–7.
- **C** (replicate=False, sensores SEM filtro): mesmo erro de binding — o
  filtro não é a causa.
- **D** (replicate=True, filtrado, USD SEM `--fix-base` + `fix_root_link=True`
  no spawn): **done, reward finito, nenhum corpo NaN** — configuração vencedora.
- **E** (replicate=False + `make_uninstanceable=True`): mesmo erro de binding
  — instancing não é a causa.

Veredito: o veneno é a junta fixa de raiz do importador (`--fix-base`) sob
replicação OvPhysX; e `replicate_physics=False` é inviável para sensores de
contato neste stack 3.0.0. A v66 aplica o modo D: conversão sem `--fix-base`,
`fix_root_link=True` no `UsdFileCfg` do robô e `replicate_physics` de volta ao
padrão (True). O portão continua o mesmo: probe finito → baseline físico →
PPO 500 iterações × 8 envs → playback → benchmark com success_rate.

### Veredito da v66 e protocolo v67 (pré-registrado em 26/10/2026)

A v66 (correção do modo D) rodou ~46 min e avançou mais que qualquer versão
anterior: probe finito ✅, baseline físico aprovado (grasp) ✅ e **o treino
PPO completou** (exit 0, checkpoints gerados) — o primeiro treino completo
do projeto. Morreu no estágio seguinte: o CLI `isaaclab play` do Isaac Lab
3.0 rejeita a flag legada `--headless` (`error: unrecognized arguments:
--headless`), então playback/vídeo e benchmark nunca rodaram e não há
success_rate ainda. Falha classificada como código/invocação, não de
hipótese nem de infraestrutura. Agravante registrado: os ramos de falha de
playback/avaliação do notebook não chamavam `save_report()`, e o
runtime-report detalhado se perdeu; corrigido na v67.

**Protocolo v67** (decisão do fundador em 26/10/2026: overhead + 64 envs):
- Treino e probe com **64 envs** (o padrão do env_cfg; a v66 forçava 8 no
  comando), mesmas 500 iterações, 24 steps/env, 5 épocas, 4 minibatches e
  seed 7: 768 mil amostras contra 96 mil da v66, com o mesmo número de
  atualizações do otimizador (10 mil). Hipótese: a T4, ociosa com 8 envs,
  entrega as 8× amostras em tempo total comparável ou menor, e a política
  aprende mais por atualização.
- Playback sem `--headless`; `save_report()` em todos os ramos de falha.
- Overhead: a fusão probe+baseline num boot só foi avaliada e adiada
  (economia <1 min contra risco de reentrada do launch_simulation não
  provada); o cache da instalação do Isaac Lab via dataset fica como
  trabalho futuro desenhado (venv de vários GB gerada por kernel construtor,
  com fallback para install completo) — payback estimado só após ~6 corridas.
- Critério de veredito: todos os estágios verdes, success_rate do benchmark
  oficial reportado, e tempo total comparado aos ~46 min da v66.

### Veredito da v67 (26/10/2026)

Pipeline inteira verde pela primeira vez: probe (64 envs) exit 0, baseline
success=True com grasp sensor aprovado, PPO 500×64 exit 0 em **421 s de
treino** (0,76 s/iteração; corrida total ~15 min contra ~46 min da v66),
playback exit 0 com vídeo gerado e benchmark oficial executado (800 steps
× 16 envs). **success_rate oficial = 0.0** (medição `benchmark_play play
success_rate` no summary; confere com o `Metrics/success_rate: 0.0000` do
treino). Reward médio do benchmark 0,738 ± 0,231; no treino, término por
`object_lost` em 100% dos episódios e reward concentrado em termos parciais
(place 0,096; grasp/lift/release = 0). Ou seja: a infraestrutura está
resolvida; o gargalo agora é aprendizado — a política perde a peça antes de
completar grasp→lift→release. Próxima frente é desenho de
reward/currículo/dificuldade, não encanamento.

O notebook quebrou DEPOIS de todos os estágios, no parsing do JSON do
benchmark: o summary é uma lista de fases com medições nomeadas, não um dict
com `success_rate` — parsing corrigido (extrator tolerante a lista/dict) com
teste de regressão; suíte 22/22. Artefatos (runtime-report, baseline,
schema/summary do benchmark e o vídeo da política) recuperados por extração
seletiva do `_output_.zip` de 16,5 GB via requisições range no diretório
central ZIP64, salvos em `outputs-v67/` — sem baixar o zip inteiro.

### Protocolo v68 — reward staged + currículo (pré-registrado em 27/10/2026)

Experimento `fanuc-reward-curriculum-v68` (issue #1; decisão em
`docs/integracoes-decisao-2026-10-27.md`). Diagnóstico pós-v67: o MDP
tornava o grasp inalcançável por exploração — termo `grasp` em
conjunção sem degraus, `place` sem trava pagando a peça parada (ótimo
local), `reach` mirando o centro da peça, contato fora das
observações, dificuldade cheia desde a iteração 0.

Mudança única = MDP + escala declarada, no padrão das tarefas
oficiais do Isaac Lab:

- `reach` mira o ponto de grasp (peça − `tool_to_part_offset`),
  kernel `1 − tanh(d/0.1)`;
- termo novo `grasp_contact` (`object_held`, peso 2, por passo):
  garra fechada + contato bilateral + peça no tool;
- `lift` binário por passo (peso 6), travado no segurar;
- `place` multiplicado pela trava `object_held`;
- observações ganham as forças de contato dos dois dedos;
- currículo de esteira em 3 fases por passo de treino (0–40% / 40–75%
  / 75–100%): velocidade (0,0) → (0,02–0,04) → (0,05–0,10) m/s,
  jitter 0 → 0,05 → 0,10;
- escala: 256 envs × 1500 iterações × 24 passos (~9,2 M amostras);
  demais hiperparâmetros idênticos à v67; benchmark oficial idêntico
  (800 steps × 16 envs, seed 7 no treino).

Veredito (impresso pelo notebook):

- **PROVA**: success_rate ≥ 0,30 no benchmark e grasp sustentado na
  fase de dificuldade cheia → reward/currículo viram o padrão da
  tarefa; frente seguinte é o BC warm-start (issue #3) mirando 0,6+.
- **PARCIAL**: grasp > 0,20 mas success < 0,30 → o gargalo migrou
  para place/release; acionar a issue #3 antes de mais escala.
- **MATA**: grasp ≈ 0 mesmo na fase 1 (peça estática) → parar de
  gastar GPU em PPO puro; BC vira o experimento principal.

Orçamento estimado: ~65–90 min de treino + overhead (~1h30–2h por
kernel). Baseline físico (scriptado) continua sendo o portão antes do
PPO: se o baseline falhar com o MDP novo, a corrida para no portão.

### Protocolo do spike Newton (pré-registrado em 27/10/2026)

Experimento `fanuc-newton-spike` (issue #2): a MESMA tarefa e o mesmo
pipeline da v67, trocando só o preset de física para
`physics=newton` (solver MuJoCo-Warp), em kernel separado
(`--physics newton --kernel-slug fanuc-newton-spike` no builder).
Baseline de comparação = v67/OvPhysX (421 s de treino, success 0.0).

Portões em ordem: (0) importação do asset e probe finito nos 64 envs,
com ganhos de atuador resolvidos conferidos (risco double-PD);
(1) baseline scriptado registrando contato bilateral pelos sensores
filtrados — sensor mudo invalida o spike; retune de contato limitado
a ≤ 3 tentativas (nconmax/atrito/impratio); (2) PPO 500×64 sem NaN,
com success_rate e tempo de parede contra a v67.

- **PASS**: estágios verdes e (success > 0 ou termos grasp/lift/
  release ativando onde a v67 zerou), tempo ≤ v67 → Newton vira
  candidato a backend padrão para a frente de aprendizado.
- **FAIL de engine**: baseline sem contato bilateral após o retune
  limitado → Newton sai de cena sem custo afundado; a linha OvPhysX
  segue como única.

### Veredito da v68 — falha no probe, sem treino (27/10/2026)

Status Kaggle `error` na versão lançada 27/10/2026 ~00:35
(`pascalcoimbra/fanuc-conveyor-pick-place`, 256 envs × 1500 it).
Pré-voo e testes locais verdes, e a corrida morreu no probe do
ambiente: baseline, PPO, playback e benchmark nunca rodaram — não
há success_rate, grasp, lift ou place para avaliar contra os
portões PROVA/PARCIAL/MATA, que ficam **não avaliados**.

Causa raiz, confirmada no código da main: `env_cfg.py` define
`finger_contact = ObservationTerm(func=mdp.finger_contact_forces)`
na classe `ObservationsCfg.PolicyCfg`, mas `mdp/__init__.py` não
exporta `finger_contact_forces` (a função existe em `mdp/cell.py`).
O import da tarefa quebra na definição da classe com
`AttributeError: module 'fanuc_cell_lab.tasks.conveyor_pick_place.mdp'
has no attribute 'finger_contact_forces'`, e o notebook aborta no
probe. O mesmo `__init__.py` também não exporta `object_held` nem
`belt_difficulty_curriculum`, que quebrariam em seguida em
`RewardsCfg`/`CurriculumCfg` — a correção é completar as
exportações, não redesenhar o MDP. Lição de teste: a suíte local
passou 26/26 porque os testes novos são baseados em string
(`assert "def finger_contact_forces(" in mdp`); falta um teste que
importe o módulo e confira os atributos usados por `env_cfg`.

Consequência para os portões: o gatilho da issue #3 **não dispara**
— PARCIAL/MATA são resultados de aprendizado medido, e aqui não
houve aprendizado algum. Nenhum relançamento pelo monitor: a
correção (exportações + teste de atributo) volta para a conversa
principal decidir antes de qualquer nova versão do kernel.

### Veredito do spike Newton — falha na conversão, portão 0 não alcançado (27/10/2026)

Status Kaggle `error` no kernel `pascalcoimbra/fanuc-newton-spike`
(lançado junto com a v68). A corrida morreu antes do probe, na
conversão URDF→USD do robô: baseline, PPO e comparação com a v67
nunca rodaram, e nenhum portão (0, 1 ou 2) foi alcançado. O spike
fica **inconclusivo** — nada aqui diz respeito ao solver
MuJoCo-Warp em si.

Causa raiz, confirmada no builder: a parametrização troca o token
`'--extra', 'ovphysx'` por `'--extra', 'newton'` em TODOS os
comandos do notebook, incluindo a conversão, que vira
`uv run --extra importers --extra newton ...` — e o uv responde
`error: Extra 'newton' is not defined in the 'optional-dependencies'
table for 'isaaclab-dev'`. Extra do uv não é token de física: o
conversor precisa manter os extras `importers`/`ovphysx`, e o
token `newton` só pertence ao argumento `physics=` e ao preset
da tarefa. A conferência do lançamento que contou
`--extra newton` como prova de parametrização correta confundiu
presença do token com correção do comando.

Próximo passo, a decidir na conversa principal: corrigir o
builder para não trocar o extra do conversor (e validar o
notebook gerado comando por comando) antes de relançar o spike;
sem isso, qualquer nova tentativa morre no mesmo ponto.

### Correção v68b / spike v2 (27/10/2026)

As duas corridas da madrugada morreram em estágios diferentes, por
bugs da preparação (não das hipóteses) — vereditos registrados no
commit b3caed3:

- **v68**: o probe importou o env_cfg e morreu com
  `AttributeError: mdp has no attribute 'finger_contact_forces'` —
  `mdp/__init__.py` mantém lista explícita de exports e as três
  funções novas da v68 ficaram fora dela.
- **Spike Newton**: a conversão URDF→USD morreu com
  `Extra 'newton' is not defined in the optional-dependencies table`
  — o builder renomeava o extra do uv para `newton`, que não existe;
  o backend é dependência padrão do workspace Isaac Lab 3.0.

Correções (branch fix/v68b-mdp-exports-newton-extra): exports
completos no `mdp/__init__.py`; builder em modo não-default REMOVE o
par `'--extra', 'ovphysx'` dos comandos em vez de renomear; testes de
regressão estruturais para os dois bugs (exports × referências do
env_cfg; ausência do extra proibido) + prova funcional da montagem
newton (physics=newton nos 4 comandos, extras de vídeo/importers
preservados). Suíte 28/28. Relançamento: v68b com o mesmo protocolo
da v68 (256×1500) e spike v2 com o mesmo protocolo do spike (64×500)
— as hipóteses e os portões não mudam.

### Vereditos v69 / spike v2 e correções (27/10/2026)

Relançamentos pós-PR #5 (exports do mdp + builder sem extra
inventado): os dois kernels avançaram um estágio e morreram no
seguinte — progresso real, hipóteses ainda não testadas.

- **v69** (`fanuc-conveyor-pick-place` v69): probe **verde** com o
  MDP da v68 (`{"finite": true, "steps": 24, "num_envs": 64}`,
  currículo ativo) — o bug de exportação morreu. O baseline morreu
  ANTES de testar física, em `make_demo_episode`:
  `ValueError: Pose IK did not converge: position residual 1.0006 m,
  orientation residual 1.8201 rad`. Causa raiz reproduzida
  localmente com o resíduo exato: o currículo da v68 já tinha
  gravado a fase 1 no env do baseline (peça estática, velocidade
  0, jitter 0 → pick em x=0,80), e o IK encadeado do professor
  diverge nesse pick — nas condições históricas (pick ≈ 0,95–1,20)
  ele converge. Correção: o baseline roda com `cfg.curriculum =
  None` — ele é o portão físico histórico, não uma medição do
  currículo. O probe segue com currículo (é a foto do treino).
- **Spike v2** (`fanuc-newton-spike` v2): a conversão morreu em
  `launch_simulation` com `ValueError: Invalid physics config:
  'newton' (expected 'physx', 'isaacsim_physx', 'newton_mjwarp',
  'newton_vbd', or 'ovphysx')` — o patch do conversor trocava o
  backend para `newton`, nome que o launcher do Isaac Lab 3.0 não
  aceita. Correção estrutural: o preset da tarefa foi renomeado
  para o nome canônico **`newton_mjwarp`** (`CellPhysicsCfg`), e o
  token do builder passa a ser `newton_mjwarp` em todos os usos
  (preset `physics=`, conversor e scripts) — um nome só, válido em
  todas as camadas.
- **Scripts**: probe e baseline tinham `args.physics = "ovphysx"`
  hardcoded — no spike eles mediriam o backend errado. Agora leem
  `FANUC_CELL_PHYSICS` (o notebook define por experimento no
  `baseline_env`; default `ovphysx`), e o relatório do baseline
  registra o backend real.

Suíte 29/29 com regressões dos três bugs. Relançamento: v68 como
v70 do kernel principal (256×1500, OvPhysX) e spike v3
(`--physics newton_mjwarp`, 64×500), portões inalterados.

### Veredito do spike v3 — portão 0 verde, portão 1 reprovado por contato (27/10/2026)

Kernel `pascalcoimbra/fanuc-newton-spike` v3, status `error` — mas
desta vez a falha é física, não de preparação: conversão, importação
e **probe passaram** (`{"finite": true, "steps": 24, "num_envs":
64}` — portão 0 ✅). O baseline scriptado rodou o episódio inteiro
sob `newton_mjwarp` e o notebook abortou no portão do baseline
(`RuntimeError: Scripted physical baseline did not complete`) com o
relatório do próprio baseline impresso no log:

- `"success": false`, `"scripted_grasp_sensor_passed": false`;
- `bilateral_contact_steps = 1`, **`closed_bilateral_contact_steps
  = 0`** — a garra nunca fechou sobre a peça com os dois dedos
  (baseline OvPhysX aprovado: sensor de grasp verde, picos
  957/768 N);
- contato **unilateral**: picos de força normal de **63,6 N (dedo
  esquerdo) × 250,8 N (direito)** — um dedo empurra, o outro mal
  encosta;
- a peça termina **ejetada** em (1,633; −0,005; 0,040) — fora da
  esteira, congelada nessa posição dos ~5 s em diante; o professor
  completa as fases no vazio.

Classificação pelo protocolo: **não é FAIL de engine ainda** — o
portão 1 prevê retune de contato limitado a ≤ 3 tentativas
(nconmax/atrito/impratio) antes de declarar o baseline sem contato
bilateral. Nenhuma tentativa havia sido gasta: a v3 rodou com a
formulação default (cone `pyramidal`, `impratio` default = 1).

**Correção / retune — tentativa 1 (spike v4)**: formulação de atrito
reforçada no preset `newton_mjwarp` (`CellPhysicsCfg`), seguindo o
guia oficial de tuning do MJWarp (cone elíptico quando um contato
válido ainda escorrega; `impratio` = impedância friccional relativa
à normal): `cone="elliptic"` e `impratio=10.0`. Capacidades
(`nconmax=512`, `njmax=1024`), integrador e a linha OvPhysX
inalterados. Teste de regressão estrutural novo (preset com o
retune presente, default OvPhysX intacto). Relançamento do spike
v4 com os portões inalterados: se o baseline seguir sem contato
bilateral fechado, restam 2 tentativas antes do FAIL de engine.

### Veredito do spike v4 — portão 1 reprovado de novo; retune tentativa 2 (spike v5) (28/10/2026)

Kernel `pascalcoimbra/fanuc-newton-spike` v4 (retune tentativa 1:
cone `elliptic` + `impratio=10`), status `error` — de novo no
portão 1, com o baseline rodando o episódio inteiro e o notebook
abortando em `RuntimeError: Scripted physical baseline did not
complete`. O portão 0 passou (`{"finite": true, "steps": 24,
"num_envs": 64}`); PPO nunca rodou. Comparação direta com a v3 —
a tentativa 1 **não teve efeito mensurável**:

- `"success": false`, `"scripted_grasp_sensor_passed": false`;
- `bilateral_contact_steps = 1` (o mesmo passo único da v3),
  **`closed_bilateral_contact_steps = 0`**;
- picos unilaterais de **63,64 N (esquerdo) × 250,84 N
  (direito)** — os 63,6 × 250,8 N da v3 na precisão impressa;
- peça ejetada para **(1,633; −0,005; 0,040)** — a mesma posição
  final da v3. Pelos snapshots de fase: em `grasp` (3,0 s) a peça
  está em x=1,040 sobre a esteira; em `hold` (3,5 s) está em
  x=1,546, z=0,057 — lançada ~0,5 m no fechamento da garra.

Leitura: a formulação de atrito (cone/impratio) regula um atrito
que nunca chega a atuar — a pinça **não se forma**. O contato
dedo×peça é um evento único e unilateral: o primeiro dedo que
encosta empurra a peça de 0,35 kg para fora do alcance do
segundo, e um contato de ponto único não oferece resistência à
rotação/translação enquanto isso. Classificação pelo protocolo:
caso (a) de novo — restam **2 tentativas** de retune antes do
FAIL de engine.

**Correção / retune — tentativa 2 (spike v5)**: atacar a
formação do contato, mantendo cone/impratio da tentativa 1.
Campos conferidos no fonte pinado do Isaac Lab (tag `v3.0.0`,
`mjwarp_manager_cfg.py`/`newton_manager_cfg.py`):

- `enable_multiccd=True` no `MJWarpSolverCfg`: pares convexos
  suportados passam a gerar um **manifold** de contatos (vários
  pontos na superfície) em vez de um ponto único; a doc do campo
  motiva exatamente "melhorar a estabilidade de contatos planos
  em stacking e grasping", e o exemplo oficial de mão do Isaac
  Lab usa a flag para contatos nos pads dos dedos;
- `nconmax=1024`, `njmax=2048`: a folga de capacidade que a
  mesma doc prescreve ao ligar o multiccd (mais pontos de
  contato e linhas de restrição por ambiente);
- `default_shape_cfg=NewtonShapeCfg(mu=1.5)` no `NewtonCfg`:
  atrito default das shapes **sem material explícito** — a cena
  da tarefa não declara nenhum — elevado do default 1.0 da
  v3.0.0 para 1.5, para a mancha de contato segurar a peça na
  direção tangencial (o esguicho é ao longo da esteira, +x) em
  vez de deixá-la escapar durante o fechamento.

Teste de regressão estrutural novo (preset com a tentativa 2,
tentativa 1 preservada, OvPhysX intacto). Relançamento do spike
v5 com os portões inalterados: sucesso do retune se mede por
`closed_bilateral_contact_steps > 0`, picos mais simétricos e a
peça terminando o episódio sobre a esteira. Se o portão 1
falhar de novo, resta **1 tentativa** — direção registrada:
rigidez/amortecimento de contato (`ke`/`kd` do
`NewtonShapeCfg`, default macio de 2,5×10³ N/m da v3.0.0) e
convergência do solver (`iterations`/`tolerance`), que a
tentativa 2 deliberadamente não tocou.

### Veredito da v70 — MATA (27/10/2026)

Kernel `pascalcoimbra/fanuc-conveyor-pick-place` v70 COMPLETO
(todos os comandos exit 0; treino 1353,6 s para 1500 it × 256 envs;
corrida total ~50 min). Primeira corrida do protocolo v68 a chegar
ao benchmark:

- probe finito ✅; baseline físico ✅ (`success: True`,
  `scripted_grasp_sensor_passed: True`, esteira 0,069 m/s, picos
  957/768 N) — o portão sem currículo validou as condições
  históricas;
- treino 1500/1500 iterações, playback ✅, benchmark ✅:
  **success_rate = 0.0** (idêntico à v67);
- termos do treino: `reach` oscilando 0,25–0,42 (a política aprende
  a pairar no ponto de grasp), mas **`grasp_contact` = 0,0 em TODAS
  as janelas logadas — inclusive na fase 1 estática**; `lift`,
  `grasp`, `place` e `release` zerados o treino inteiro;
  `Metrics/success_rate` 0,0000. Terminações: object_lost ~45,8%,
  time_out ~54,2%, success 0%.

Classificação pelos portões pré-registrados: **MATA** — grasp ≈ 0
mesmo na fase estática, com 9,2 M amostras. A conjunção do
`object_held` (garra fechada ∧ contato bilateral ∧ perto do tool)
não foi descoberta pelo PPO puro nem com alvo parado, reward em
estágios e currículo. Consequência do protocolo: **parar de gastar
GPU em PPO puro; o BC warm-start (issue #3) vira o experimento
principal.** A branch `frente/bc-warmstart-prep` está pronta
(coletor do professor, portão de demos ≥ 0,80, suíte 33/33); o
merge é a decisão que destrava o lançamento.
### Protocolo do experimento BC — warm-start do professor (pré-registrado em 27/10/2026)

Experimento `fanuc-bc-warmstart` (issue #3). Preparação mergeada antes
do veredito da v68 (coletor, BC em numpy, entrada de treino
warm-start) para o lançamento ser imediato quando o gatilho disparar.

**Gatilho** (veredito da v68, seção "Protocolo v68" acima): PARCIAL ou
MATA. Com PROVA da v68, o BC roda na rodada seguinte mirando 0,6+,
com baseline = success da v68.

**Hipótese:** semear o ator com behavior cloning das demonstrações
do professor scriptado (que completa o ciclo grasp→lift→place no
baseline físico desde a v51) coloca o PPO dentro da região de grasp
desde a iteração 0; com o reward staged da v68 segurando o
comportamento, o success_rate supera o da v68 na mesma escala.

**Pipeline do kernel** (builder `--bc-warmstart`, slug separado
`fanuc-bc-warmstart`; o kernel principal não muda):
1. Portões de sempre: probe anti-NaN → baseline físico aprovado.
2. Coleta: `scripts/collect_demos.py`, 200 episódios sequenciais do
   professor (seeds 1000+), pares (obs do grupo policy, ação) em
   `.npz` via `fanuc_cell.demos`. Limitação v1 registrada: coleta
   1-env sequencial (~385 passos/episódio); professor vetorizado é
   trabalho futuro, não deste experimento.
3. Portão das demos: success do professor ≥ 0,80 na coleta
   (`pretrain_bc.py` exit 3 abaixo disso) — demo ruim não vira BC.
4. BC: `fanuc_cell.bc` (MLP [256,128,64] ELU, espelho do ator),
   300 épocas, MSE nas ações; relatório com perda inicial/final.
5. Portão de fumaça da entrada warm-start: o treino usa
   `scripts/train_ppo_bc.py` (runner RSL-RL padrão + injeção dos
   pesos por ordem de camadas, marcador `BC_WARMSTART_APPLIED` no
   log com shapes conferidos; divergência de arquitetura aborta
   ANTES do learn). Primeira corrida BC que não imprimir o marcador
   = experimento inválido, não veredito.
6. PPO idêntico à v68 (256 envs × 1500 it, seed 7, mesmo benchmark
   oficial) — a única diferença é o ponto de partida do ator.

**Métricas:** primária = success_rate no benchmark oficial.
Secundárias: grasp/lift/release do treino (o warm-start deve mostrar
grasp > 0 desde as primeiras iterações — é a assinatura do
mecanismo), perda do BC, taxa de contato sustentada no fine-tune
(degradação da "gramática de contato" = alerta registrado na issue).

**Veredito** (baseline S68 = success_rate oficial da v68):
- **PROVA**: success ≥ max(0,30; S68 + 0,10) → warm-start vira o
  ponto de partida padrão das próximas corridas; frente seguinte:
  DAgger ou escala, mirando 0,6+.
- **PARCIAL**: success > S68 mas abaixo da faixa de PROVA, ou grasp
  sustentado sem place → o gargalo migrou; iterar nos estágios
  finais do reward antes de mais BC.
- **MATA**: success ≤ S68 (ou grasp ≈ 0 mesmo com warm-start) → o
  problema não é exploração: é observabilidade/capacidade (o
  professor usa estado interno não observável — âncora de captura,
  máquina de fases). Próxima frente vira estudo de observações/
  memória (ator recorrente ou obs enriquecida), não mais demos.

Orçamento estimado: coleta ~15–35 min (sequencial) + BC ~1 min +
treino ~1h30–2h = kernel de ~2–2,5 h.

### Investigação da divergência IK — coleta BC v1 (27/10/2026)

A v1 do kernel BC morreu na coleta, episódio 76/200 (seed 1075),
com `Pose IK did not converge` (resíduo 0,0068 m / 0,0080 rad) no
re-solve contra a peça medida. Investigação pedida pelo fundador
antes de qualquer relançamento:

- **Não há divergência de código entre baseline e coletor**: os dois
  drivers têm o drive do professor linha a linha igual (mesma
  CellConfig, mesmo re-solve com seed nas juntas de referência,
  mesma âncora de captura). A diferença é amostragem: o baseline
  roda UMA condição (seed 23: start 0,832, vel. 0,069, pick 1,039 —
  aprovada desde a v51), o coletor varre 200 condições sorteadas
  (start ∈ [0,80; 0,90], vel. ∈ [0,05; 0,10]).
- **Reprodução local (núcleo puro)**: com start 0,80 / vel. 0,05
  (pick 0,95), o primeiro re-solve do approach (alvo = peça medida
  − offset + clearance = (0,795, 0, 0,37), seed = juntas do waypoint
  approach) trava com resíduo estacionário 0,0507 m — idêntico em
  240 ou 5000 iterações e insensível ao damping: é **mínimo local do
  DLS dependente da seed**. Da seed zero (de onde a cadeia de
  construção do episódio parte), o mesmo alvo converge a 0,07 mm —
  o alvo é alcançável; não é fronteira de workspace (na altura do
  grasp, x até 1,105 converge da seed de referência).
- **Causa raiz**: o re-solve de tracking semeado nas juntas de
  referência é frágil numa região do approach (alvo atrás do
  waypoint de referência por até `vel × 1,2 s`, com clearance); o
  baseline nunca expôs a armadilha por rodar uma única seed. O
  resíduo menor da corrida real (6,8 mm) é a mesma bacia sob alvos
  perturbados pela física (a condição exata do episódio 76 depende
  do RNG do Isaac e não é reproduzível localmente).

**Correção (branch `fix/ik-coletor-bc`, suíte 41/41)**:
1. `inverse_kinematics_pose` ganha `fallback_seeds` (default vazio =
   comportamento bit-idêntico): em falha da seed primária, tenta as
   reservas em ordem e levanta o erro original se todas falharem.
2. Baseline e coletor passam `fallback_seeds=(zeros,)` no re-solve.
3. Coletor resiliente e honesto: episódio que abortar mesmo com
   fallback é descartado (`DemoRecorder.register_failed_episode` —
   pares parciais fora do BC, episódio dentro do denominador do
   portão ≥ 0,80) e a coleta segue; o resumo reporta
   `ik_failed_episodes`.

Relançamento do kernel BC a partir desta branch (após merge) é a
decisão seguinte do fundador.

### Análise da coleta BC v3 — por que o professor faz 0,515 (27/10/2026)

Atribuição por episódio a partir do `bc-demos.npz` da v3
(trajetórias reconstruídas: TCP por FK das juntas gravadas, peça =
TCP + observação relativa; 200 episódios, 89.000 passos):

- **103 success (51,5%), 111 grasped (55,5%)**. Das 89 falhas de
  grasp: **75 (classe D)** — captura e contato bilateral acontecem,
  mas no lift a garra fecha até **0,0120 m (o setpoint cheio, vazia)**
  e a peça fica na esteira no ponto de pick (final x≈1,11, z=0,15),
  contato 0 N no fim do lift; **14 (classe C)** — contato **zero** o
  episódio inteiro, garra fecha vazia, peça largada no pick; classes
  "sem janela geométrica" e "peça não parou" = **0**. Nos 111
  grasped, a garra segura em **0,0060 m** (a peça entre os dedos) e o
  contato sustenta ~12 N até o fim do lift.
- **Dependência da condição**: grasp 0,85 com velocidade
  0,06–0,075 m/s cai para **0,35** em 0,075–0,09; pick 1,02–1,10 →
  grasp 0,64 contra **0,32** em 1,10–1,17. O baseline (seed 23:
  start 0,832, vel. 0,069, pick 1,039 — números do runtime-report
  da v3) mora na faixa boa: o portão de condição única nunca amostrou
  a região ruim.
- **Mecanismo**: a captura exige pinch (≥4 mm) + TCP a <0,16 m, mas
  a garra só começa a fechar em t=3,0 s, junto com a descida sem
  clearance — a peça segue em movimento cinemático durante
  descida+fechamento e avança `vel × atraso`; em velocidade/pick
  altos ela já não está centrada entre as garras quando a esteira
  pausa. A âncora congela a posição desalinhada e o lift sai vazio.
  Forças de centenas de N no grasp ocorrem também nos sucessos
  (baseline: picos 957/768 N) — esmagamento transitório contra o
  drive cinemático é normal nesta célula; o discriminante é o que
  sobra no lift (contato sustentado + garra em 6 mm × garra vazia
  em 12 mm + peça no pick).
- Dos 8 grasped sem success, a maioria **derruba a peça no
  transfer** (finais em y −0,69…−1,40, z 0,04 — além do pedestal):
  vazamento secundário de ~4 pp, mesma família (pinch firmeza).
- **Correção indicada** (decisão do fundador): grasp em dois tempos
  — pairar sobre o pick com a garra pré-fechada no limiar do pinch
  para a captura disparar com a peça centrada e a esteira pausada,
  e só então descer os últimos centímetros e fechar sobre a peça
  PARADA, em vez de descer+fechar sobre a peça em movimento.
  Relançar a coleta sem essa correção reproduz o mesmo portão.

### Correção: captura por proximidade + grasp em dois tempos (27/10/2026)

Duas mudanças a partir da análise da coleta BC v3, aplicadas juntas e
pré-registradas antes de qualquer nova corrida:

1. **Captura por proximidade da estação** (`advance_conveyor`): a
   esteira agora solta a peça para a física assim que o TCP entra no
   raio de captura de 0,16 m (`capture_attempt = close_to_pick`), sem
   exigir o pinch prévio. A regra antiga era o gargalo: a garra só
   começa a fechar na descida, então a peça seguia em movimento
   cinemático enquanto os dedos fechavam sobre ela. Comportamento de
   estação real: a peça para quando o robô está em posição.
2. **Grasp em dois tempos do professor** (`teacher_gripper_action`,
   nos dois drivers, baseline e coletor): na fase grasp a garra fica
   ABERTA até a captura parar a peça; só então fecha (setpoint de
   12 mm). Do hold em diante fecha de qualquer jeito — captura que
   nunca dispara degrada para o comportamento antigo, sem travar o
   episódio.

**Protocolo BC v4** (idêntico ao protocolo BC, sob a semântica
corrigida): probe + baseline + coleta de 200 demos (seeds 1000+) →
portão do professor ≥ 0,80 → BC 300 épocas → PPO warm-start
256×1500 com marcador `BC_WARMSTART_APPLIED` → benchmark oficial.
Veredito contra S68 = 0,0: PROVA ≥ 0,30 / PARCIAL > 0 / MATA ≤ 0.
Ressalva registrada: v4 roda sob a regra de captura nova — o portão
do professor é comparável dentro da mesma versão do env, e um
eventual PROVA deve ser lido junto desta mudança de tarefa.

### BC v4: portão aprovado, PPO morreu na config — correção e protocolo v5 (27/10/2026)

Veredito da v4 por estágio: probe ✅, baseline ✅, coleta 200/200
(ik_failed 0), **professor success 0,865 / grasp 0,92 — portão ≥ 0,80
APROVADO** (a correção captura+dois tempos funcionou; v3 era 0,515);
BC ✅ (300 épocas, loss 0,01063 → 0,000157). O PPO **nunca começou**:
`train_ppo_bc.py` construía o runner com `agent_cfg.to_dict()` cru e
o rsl-rl ≥ 4 (Isaac Lab v3.0.0 pede rsl-rl-lib ≥ 5.0.1) morreu com
`KeyError: 'class_name'` em `resolve_class(cfg["actor"])`. Sem
marcador, sem benchmark: **experimento inválido no estágio PPO**,
hipótese BC aberta com professor aprovado.

Causa: o treino oficial (`isaaclab train`) chama
`handle_deprecated_rsl_rl_cfg(agent_cfg, version("rsl-rl-lib"))`
antes do runner — é ele que converte a cfg antiga (`policy`) em
`actor`/`critic` com `class_name`. O script BC pulava esse passo.

Correção (suíte 43/43): `train_ppo_bc.py` agora (1) aplica
`handle_deprecated_rsl_rl_cfg` com a versão instalada, igual ao
treino oficial; (2) a injeção BC resolve os Lineares do ator nas duas
APIs (`alg.actor.mlp` no rsl-rl novo; `alg.actor_critic.actor` no
antigo), restrita ao `mlp` quando ele existe.

**Protocolo BC v5**: idêntico ao v4 (mesma coleta, portão, BC, PPO
256×1500, benchmark; vereditos contra S68 = 0,0 inalterados), só com
o script de treino corrigido.

### BC v5: morta no manifesto do builder — correção e protocolo v6 (27/10/2026)

A v5 morreu em ~5 min, **antes de qualquer física**: o PR #7 (logger
Rerun, issue #6) adicionou `src/fanuc_cell/rerun_log.py` e o
`__init__` passou a importá-lo, mas o manifesto explícito de arquivos
do `bundled_project()` no builder não acompanhou — o zip do kernel
ficou sem o módulo e o estágio de testes locais abortou com
`ModuleNotFoundError: fanuc_cell.rerun_log`. Lição registrada: arquivo
novo em `src/fanuc_cell/` exige entrada no manifesto do builder.

Correção: `rerun_log.py` e `tests/test_rerun_log.py` no manifesto +
teste de completude (todo módulo de `src/fanuc_cell/` em disco tem
que estar no zip) + pré-voo novo: rodar o pytest **sobre a árvore
extraída do zip embutido**, reproduzindo o estágio local do kernel
(17/17). Suíte 48/48. **Protocolo BC v6**: idêntico ao v5.

### BC v6: a cadeia inteira rodou — mas fora do protocolo e sem benchmark (27/10/2026)

Por estágio: testes locais ✅ (manifesto corrigido), probe ✅,
baseline ✅, coleta 200/200 ✅ (professor **0,865 / grasp 0,92**,
portão aprovado), BC ✅ (loss 0,01063 → 0,000157), **PPO warm-start
TREINOU até o fim (exit 0)** — primeira vez que a cadeia completa
executa. Depois: **playback morreu** em
`--checkpoint latest` (`ValueError: No compatible manifested run`):
o entrypoint unificado de play do Isaac Lab 3.0 resolve o checkpoint
por um manifesto que só o `isaaclab train` oficial escreve; o nosso
runner direto não escreve. Benchmark nunca rodou.

Duas ressalvas que invalidam a v6 como experimento do protocolo:
1. **Escala errada**: o comando de treino saiu com
   `--num_envs 64 --max_iterations 500` (defaults do builder) em vez
   dos 256×1500 pré-registrados — as builds BC não passaram as flags
   `--train-num-envs/--train-max-iterations`. Amostras: 0,77 M, não
   9,2 M.
2. **Marcador não verificável**: `BC_WARMSTART_APPLIED` é impresso
   no início do stdout do treino, mas o runtime-report guarda só a
   cauda de 12k — o marcador ficou fora do registro (a injeção em si
   não levantou erro: o treino só começa depois dela). Nas métricas
   finais do treino (it. 499): success 0,0 e grasp 0,0 — sob escala
   reduzida, inconclusivo sobre o warm-start.

**Protocolo BC v7** (correções): build com 256×1500 explícitos;
playback e benchmark com o caminho EXPLÍCITO do checkpoint (glob que
o notebook já faz), sem `--checkpoint latest`; marcador do warm-start
também gravado em arquivo JSON nas saídas. Todo o resto idêntico.

### Protocolo de benchmark multi-seed (B3, issue #10 — 27/10/2026)

A partir daqui, todo veredito oficial de benchmark reporta **média ±
desvio de 3 seeds fixas (101, 202, 303)** do `isaaclab benchmark play`
(16 envs × 800 passos cada): `policy_success_rate` no runtime-report
passa a ser a média, com `policy_success_rates` (lista) e
`policy_success_rate_std` ao lado. Seed única não separa sinal de
ruído — os vereditos até a BC v7 foram todos de seed única (101) e
devem ser lidos com essa ressalva. A corrida definitiva usa este
formato.

**Refinamento B3 (28/10/2026, antes do primeiro veredito neste
formato — nenhum veredito emitido é afetado):** a matemática da
agregação mora num módulo puro e testado,
`fanuc_cell.benchmark_stats` (`summarize_rates`), usado pela
célula de benchmark do notebook — fonte única, sem conta inline.
O desvio passa a ser o **amostral (ddof=1)** e o runtime-report
ganha `policy_success_rate_ci95` (meia-largura do IC95 da média,
z × desvio/√n) e `policy_benchmark_seeds` (a lista usada); cada
entrada de `policy_evaluation_runs` carrega também o
`success_rate`/`grasp_rate` da sua seed. O veredito reporta as
duas métricas: além do success, o benchmark agrega o **grasp**
por seed — `Metrics/grasp_rate` (fração dos episódios encerrados
com a peça segurada ao menos uma vez), logado pelo termo de peso
0.0 `ObjectGraspRate`, espelho do canal do success — em
`policy_grasp_rates` (lista, `null` por seed quando ausente),
`policy_grasp_rate` (média), `policy_grasp_rate_std` e
`policy_grasp_rate_ci95`; se o benchmark não reportar grasp,
`policy_grasp_rate` fica `null` e o success segue como portão
único (a ausência é registrada, nunca silenciosa). As seeds são
parâmetro do builder (`--benchmark-seeds`, default
`101,202,303` = este protocolo); uma seed só reproduz a leitura
legada (desvio/IC95 = 0). Formato da linha de veredito:
`success_rate 0,000 ± 0,000 (IC95 ± 0,000, n=3)` — idem grasp.

### Estudo de observabilidade/memória (issue #8) e Experimento 1 (27/10/2026)

Estudo completo em `docs/estudo-observabilidade-2026-10-27.md`.
Conclusão principal: a hipótese com mais suporte **não é
observabilidade** — é **destruição do warm-start na interface
BC→PPO**: `grasp` em 0,0 desde o início do treino da v7 indica que o
comportamento clonado nunca se manifestou nos rollouts do PPO
(mecanismos documentados: ruído de exploração inicial, bônus de
entropia no fine-tuning, crítico aleatório gerando vantagens-lixo,
mismatch de normalização, ausência de âncora de imitação — o
antídoto canônico é DAPG: imitação continuada durante o RL).

**Experimento 1 (protocolo pré-registrado; um kernel, dois
estágios)**: (A) rollout determinístico do **ator BC puro** em 200
episódios (benchmark do ator sem PPO); (B) **PPO warm-start
protegido** no budget da v7 (256×1500): entropia 0 (o std inicial
ja e 0,1 no cfg do projeto — abaixo do teto de 0,15 do estudo —
e fica como esta),
normalização de obs idêntica à do BC, crítico aquecido com o ator
congelado por 50 iterações. Benchmark multi-seed (B3).
Portões: **PROVA** = benchmark ≥ 0,30. **PARCIAL** = sucesso em
(0; 0,30), ou grasp final ≥ 0,20 sem sucesso, ou grasp alto que
decai (→ próximo passo: DAPG). **MATA** com leituras obrigatórias:
grasp@iteração-0 alto que colapsa confirma esquecimento (→ DAPG);
grasp do Estágio A < 0,05 condena o BC em si (→ estado
ampliado/DAgger), não o PPO. Aguardando decisão do fundador.

**Correção de implementação do Estágio B (27/10/2026)**: o
mapeamento da fonte do rsl-rl-lib 5.5.1 achou um off-by-one no
`learn()` em duas chamadas — `current_learning_iteration` guarda o
índice da última iteração concluída, então sem ajuste a iteração 49
(o fim do warmup) reexecutaria já descongelada e a corrida fecharia
em model_1498. Correção: `current_learning_iteration += 1` entre as
chamadas (50 congeladas exatas, 1500 updates, model_1499). A v8 do
kernel chegou a ser lançada sem a correção e foi substituída pela
v9 minutos depois, ainda no estágio de coleta — nenhum resultado da
v8 é válido nem será lido como experimento.

### Experimento OBS-2 — histórico de observações (issue #8, parte 2 — 27/10/2026)

Estudo completo em `docs/estudo-observabilidade-exp2-2026-10-27.md`.
Desenho: a **menor** mudança de memória primeiro — frame stacking
nativo do Isaac Lab (grupo `policy` com `history_length=8`,
achatado; 24 → 192 dims; janela de 0,16 s a 50 Hz), que ataca o
aliasing temporal sub-segundo (dinâmica do contato, latch de
captura) sem tocar MDP, reward, loop de treino ou arquitetura
(exceto a camada de entrada, desvio forçado registrado). Opções
maiores (features derivadas de sensores, crítico assimétrico, ator
recorrente) estão desenhadas no estudo como OBS-3+, não
implementadas.

**Fiação (implementada, atrás de flag, default intocado):**
variável `FANUC_CELL_OBS_HISTORY` lida no `env_cfg` (default 1 =
frame único: nenhum atributo de história é tocado, tarefa byte a
byte a de sempre) e flag `--obs-history N` no builder, que injeta a
variável no processo do notebook e no `baseline_env` (coleta, eval,
treino, playback e benchmark com o MESMO N por construção) e grava
`obs_history` no runtime-report.

**Protocolo pré-registrado:** a corrida OBS-2 só roda com o
veredito do Experimento 1 lido e o ramo da árvore do estudo
apontando para ela (PARCIAL-ii ou MATA-B do Exp. 1; ou decisão do
fundador no ramo PROVA). Kernel novo (slug `fanuc-obs2-history`):
build `--bc-warmstart --bc-exp1 --obs-history 8`, 256×1500,
benchmark multi-seed B3 — **única diferença vs. o Exp. 1 é a
história**. Portões: **PROVA** = benchmark ≥ 0,30. **PARCIAL** =
sucesso em (0; 0,30), ou grasp final ≥ 0,20 sem sucesso, ou
`grasp_rate_BC(H=8) ≥ 0,30` com colapso no PPO (→ DAPG sobre a obs
com história). **MATA** = sucesso 0, grasp final ≤ 0,02 e
`grasp_rate_BC(H=8) < 0,05` (janela curta rejeitada como causa
primária; volta à decisão do fundador entre features/crítico
assimétrico/recorrência). Demos e checkpoints não são
intercambiáveis entre valores de H.
### Protocolo do probe do limiar da garra (B6, issue #13 — pré-registrado em 27/10/2026)

> Protocolo `fanuc-gripper-threshold-b6`.

A ambiguidade, medida no código: `object_held` e `object_grasped`
declaram a garra **fechada** com as duas juntas `>=
closed_joint_position` (default **0,004 m**, parâmetro) e o baseline
scriptado usa o mesmo literal; `object_at_goal` declara a garra
**aberta** com as duas juntas `< 0,005 m` — literal solto, sem
parâmetro. Na banda [0,004; 0,005) a mesma pose é fechada para o
grasp e aberta para o sucesso, simultaneamente. Nenhum dos dois
valores foi medido: um veio do código, outro dos testes antigos.
Este protocolo mede onde a física põe o pinch e fixa UM valor.

**Medição** (kernel `pascalcoimbra/fanuc-probe-gripper-threshold`,
builder `scripts/build_probe_notebook.py`, probe
`scripts/probe_gripper_threshold.py`; nada roda na VM): cenário
`pinch` = episódio do professor com a construção do baseline, seeds
**23, 24, 25**; cenário `empty` = braço em home, fechamento até o
platô (3 s) e abertura (2 s). Cada passo grava junta esquerda/direita
da garra, força normal dos dois dedos, fase e distância peça–TCP. As
amostras cruas são salvas (`gripper-threshold-samples.json`) — a
análise (`fanuc_cell.gripper_threshold`, pura) pode ser refeita sem
nova simulação.

**Quantidades medidas.** Verdade física do pinch = contato bilateral
(forças > 0,01 N, o limiar do próprio MDP). (i) `pinch_p05`:
percentil 5 da junta MÍNIMA nos passos de contato bilateral
sustentado (>= 5 passos consecutivos) das fases grasp/hold/lift/
transfer/place — teto do limiar de fechamento. (ii)
`contact_loss_p05`: percentil 5 da junta MÁXIMA no passo de perda de
contato na abertura (último passo com contato seguido de >= 10 sem
contato) — teto do limiar de abertura. (iii) platô do fechamento
vazio (mediana da junta mínima nos 20% finais do `empty_close`).
(iv) varredura FP/FN de cada candidato da grade 0,0020–0,0080 m
(passo 0,00025) contra o contato medido. (v) ocupação da banda
atual: fração de passos com min >= 0,004 e max < 0,005, no total,
com contato e sem contato.

**Regra de leitura (fixada antes da corrida).** Teto admissível =
min(`pinch_p05`, `contact_loss_p05`); **T\*** = maior ponto da grade
sob o teto. **PROVA**: T* >= 0,0045 — adotar T* como valor único
(fechada := min >= T*; aberta := max < T*, o MESMO T* nos dois
predicados; a banda some por construção). **PARCIAL**: 0,004 <= T* <
0,0045 — manter 0,004 como valor único e alinhar só o literal de
abertura a ele. **MATA**: T* < 0,004 ou grade vazia — nem o valor do
código se sustenta: pinches reais vivem abaixo dele ou o contato
persiste abaixo dele na abertura; escalar a revisão do desenho da
detecção de grasp (a junta sozinha não separa os estados).
**SEM_SINAL**: sem pinch sustentado ou sem perda de contato medida
— a medição não aconteceu; relançar o probe, nada se decide.

**Alinhamento (commit seguinte ao veredito, PR próprio):** aplicar
o valor decidido em `object_held`/`object_grasped`
(`closed_joint_position`), no literal de abertura de
`object_at_goal` (que passa a usar o mesmo valor), no literal do
baseline (`run_scripted_demo.py`) e na asserção do teste de scaffold
— código e testes alinhados ao valor medido, como pede a aceitação
da issue #13. Até lá, código e testes seguem como estão: este PR
entrega o instrumento e a regra, não o valor.
## Modo eval-only (issue #11, B2) — implementado em 27/10/2026

Avaliar checkpoint existente sem repetir treino: o checkpoint é
empacotado como dataset Kaggle (`scripts/package_checkpoint_dataset.py`
gera a pasta com o .pt de nome preservado, `dataset-metadata.json` e
`checkpoint-manifest.json` com sha256/bytes/origem; upload pelo
`create-dataset` do conector custom.kaggle) e o builder ganha o modo
`--eval-only --eval-checkpoint <dataset>/<arquivo>.pt`:

- O builder anexa o dataset ao kernel (`dataset_sources` no metadata;
  o push reescreve o dono com `--usuario`, como já faz com o id).
- Na célula principal, o bloco de treino é substituído pela resolução
  do checkpoint em `/kaggle/input/<dataset>/<arquivo>.pt`, com portão
  de existência; playback e benchmark multi-seed (B3) usam
  `--checkpoint eval_checkpoint` — **nada de `latest`** (lição da
  BC v6) nem de glob por mtime. Probe de NaN e baseline scriptado
  (sanidade) são mantidos; o runtime-report registra `eval_only`,
  o caminho e os bytes do checkpoint.
- Validações do builder: `--eval-only` exige checkpoint explícito
  relativo a /kaggle/input, terminando em `.pt`, sem componente
  `latest`; incompatível com `--bc-warmstart`. Compõe com
  `--cinematic-video` (o caso da issue #9: vídeo de checkpoint já
  treinado) e com o passe de física. Sem as flags, o notebook gerado
  é idêntico ao de sempre (células e conteúdo do zip conferidos
  contra a main).
- Protocolo da 1ª corrida: montar com slug próprio (ex.:
  `fanuc-eval-only`), avaliar o checkpoint final da corrida BC
  vigente e conferir (a) tempo total de kernel < 8 min até o fim do
  benchmark e (b) `policy_success_rates` por seed idênticos aos da
  corrida original. Até essa corrida, a aceitação está implementada
  mas não verificada em voo.

### Protocolo do vídeo industrial — newton_rtx + cena cinematic (issue #9, pré-registrado em 27/10/2026)

Auditoria em `docs/audit-video-cena-2026-10-27.md` (Rota 2). O
código da fatia 1 já está na main (`a14c799`): flag
`--cinematic-video` no builder (só o playback troca
`--viz newton_gl` → `newton_rtx` e ganha o extra `ovrtx`;
`FANUC_CELL_CINEMATIC=1` só no env do subprocesso do play) e a
variante `CellSceneCfgCinematic` no `env_cfg` (materiais PBR,
set visual-only sem `collision_props`, luz-chave, câmera nova —
selecionada no `__post_init__` sob a variável; MDP, física e a
tarefa default byte a byte intactos, pinados em
`tests/test_isaaclab_scaffold.py`, incluindo a combinação
cinematic × física `newton_mjwarp`). Este protocolo registra a
**prova no Kaggle**, que é o aceite da issue:

1. **Pré-voo** (no próprio kernel, antes do playback): confirmar
   contra a fonte pinada v3.0.0 que o extra `ovrtx` resolve no
   workspace e que `isaaclab play --viz newton_rtx --video` grava
   offscreen. Se o nome do extra divergir da fonte, a correção é
   no builder + teste, e a corrida para aqui.
2. **Smoke de render**: playback curto (~40 frames) medindo
   s/frame no T4 e extrapolando os 400 frames. Estimativa da
   auditoria: 5–40 min. Se a extrapolação passar de ~40 min,
   reduzir para 720p no recorder antes da corrida cheia —
   decisão com o número do smoke, não no escuro.
3. **Corrida definitiva**: kernel construído com
   `--cinematic-video`, mesma tarefa e mesmo checkpoint de um
   playback `newton_gl` de referência (fisicamente idêntico por
   construção: só viz + variante de cena mudam).
4. **Aceite**: `policy_playback_exit = 0`; mp4 exportado
   renderizado em `newton_rtx` com a cena cinematic; **tempo de
   render medido e registrado** aqui como veredito, ao lado do
   tempo do playback de referência. Falha confinada ao estágio de
   playback (extra/classe de cfg) não contamina treino/benchmark
   — os estágios anteriores do mesmo kernel seguem válidos.

Fora deste protocolo (fatia 2, issue futura): correia/rolos
animados sincronizados à peça, trajetória de câmera, patch PBR
no USD do robô.

### Cache da instalação do Isaac Lab (B4, issue #12 — lado do código, 27/10/2026)

Desenho completo em `docs/isaac-cache.md`. Entregue no código:
núcleo puro `fanuc_cell.isaac_cache` (manifesto formato 1 amarrado
ao ref `v3.0.0`, descoberta do mount, plano de restauração como
dados); builder principal com `--isaac-cache-dataset` opt-in
(metadata ganha `dataset_sources`; a célula de instalação tenta
restaurar + sonda `import isaaclab, torch` e cai no clone+sync
original em qualquer falha, registrando `runtime_report['isaac_cache']`
e `timings_seconds['isaac_setup']`); kernel construtor
`fanuc-isaaclab-cache-builder` (`scripts/build_cache_notebook.py`:
sync com `UV_LINK_MODE=copy`, extras `ovphysx/video/importers/ovrtx`
aquecidos) + `scripts/prepare_isaac_cache.py` (snapshot no layout
do dataset + `--verify` pelo mesmo portão do consumidor). Sem a
flag, células e metadata gerados são idênticos aos de sempre — só
o zip do bundle ganha os arquivos (inertes) do núcleo do cache. **Pendente de
corrida** (aceitação da issue): rodar o construtor, criar o dataset
`fanuc-isaaclab-cache` da saída dele e medir a primeira corrida
completa restaurada contra a linha de base clone+sync.

### Protocolo do spike B5 — probe+baseline num boot único (issue #14, pré-registrado em 27/10/2026)

Desenho: `scripts/probe_and_baseline.py` roda os dois estágios NO
MESMO processo via `runpy` — cada estágio executa o seu arquivo
intacto, como `__main__`, com o próprio argv — ou seja, duas
entradas sequenciais em `launch_simulation` sob um único boot do
Isaac Sim. Os estágios separados não mudam em nada. O builder ganha
a flag opt-in `--single-boot-spike`, que troca os dois comandos do
kernel por um só e registra `single_boot_spike` no runtime-report
(exit de cada estágio + veredito), mantendo derivadas as chaves
atuais (`ppo_nan_probe_exit`, `scripted_baseline`) e os dois
portões (exit != 0 aborta; gate grasp/place do baseline idêntico).
A flag não combina com `--bc-warmstart` (spike isolado, protocolo
próprio).

Protocolo: kernel da linha principal construído com
`--single-boot-spike` (mesma física e parâmetros dos boots
separados: probe 64 envs/24 steps/seed 7, baseline seed 23) e
comparação direta com a corrida de boots separados da mesma versão
de código. **ADOTA** se, dentro do mesmo processo, o probe sair 0
(finite) e o baseline produzir o mesmo veredito (success +
scripted_grasp_sensor_passed). **MATA** se o segundo
`launch_simulation` falhar, travar ou divergir — o spike encerra
sem custo e os boots separados ficam. A economia esperada é
<1 min/corrida: o que está em julgamento é a reentrada, não o tempo.

Status: implementação e testes estruturais prontos em 27/10/2026
(suíte verde: 69/69 na base da branch e 71/71 após o merge da
main, incluindo o núcleo de orquestração exercitado com
estágios falsos — SystemExit, exceção e argv conferidos). O
veredito aguarda a corrida do spike no Kaggle: nenhuma simulação
roda na VM, e a reentrada só se prova com o Isaac de verdade.
