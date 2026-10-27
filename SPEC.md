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
