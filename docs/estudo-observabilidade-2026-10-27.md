# Estudo — Issue #8: Observabilidade/Memória (FANUC × NVIDIA)

**Pergunta:** por que a política não pega a peça — a observação de 24 dims num frame único
não identifica o grasp, ou o problema está em outro lugar?
**Fatos de partida (não re-derivados):** obs = 24 dims (juntas 8, vel. juntas 8, peça rel.
TCP 3, vel. peça 3, forças de contato dos 2 dedos 2); ator MLP feedforward [256,128,64]
RSL-RL PPO; professor scriptado age por **fases** com **estado de captura** e tem sucesso
0,865; PPO puro = 0,0 em 9,2M amostras; BC warm-start → PPO apaga a imitação, grasp 0,0
**do início ao fim**, benchmark 0,0.

---

## 1. Diagnóstico fundamentado (hipóteses ranqueadas por suporte)

### H1 — Destruição do warm-start na interface BC→PPO (otimização) — **mais suporte**

A assinatura decisiva está no fato estabelecido: **grasp 0,0 desde o início do PPO**.
Se os pesos BC estivessem sendo expressos nos rollouts, o grasp das primeiras iterações
seria ≈ grasp do ator BC em rollout (esperado ≫ 0, dado professor 0,865). Grasp zero
desde o começo indica que o comportamento clonado **nunca se manifestou** — ou foi
destruído nas primeiras atualizações. Mecanismos conhecidos, todos plausíveis aqui:

1. **Ruído de exploração na inicialização.** O PPO amostra ações com desvio inicial
   (`init_noise_std`; no RSL-RL clássico o default é 1.0). Um ator BC determinístico
   cercado de ruído gaussiano dessa magnitude não executa a trajetória clonada: os
   rollouts iniciais já são de uma política ruidosa, e as vantagens empurram para longe.
2. **Bônus de entropia no início do fine-tuning.** O FLaRe (arXiv:2409.16578) documenta
   exatamente este colapso: ao fine-tunar política pré-treinada com PPO, o termo de
   entropia "pode distorcer rapidamente o gradiente no início e causar o desaprendizado
   da política pré-treinada" — eles o **removem**. Remover qualquer um dos componentes
   de estabilização deles resulta em colapso de treino.
3. **Crítico aleatório.** Vantagens iniciais calculadas por um crítico frio são ruído;
   os primeiros gradientes do ator são, portanto, destrutivos. A literatura de
   fine-tuning online mostra que crítico incompatível com a política pré-treinada causa
   queda inicial de desempenho (comparações de inicialização de Q, arXiv:2505.16856).
   Antídoto padrão: aquecer o crítico (ator congelado/LR≈0) nas primeiras iterações.
4. **Mismatch de normalização de observação.** O BC treina sobre obs brutas; o modelo
   do PPO tem normalização própria (`obs_normalization` no `RslRlMLPModelCfg`/
   `RslRlRNNModelCfg`, default False no wrapper atual — mas estatísticas empíricas do
   runner, quando ativas, começam do zero). Qualquer divergência de escala entre o que
   o BC viu e o que o ator recebe no PPO deforma a função clonada silenciosamente.
5. **Ausência de âncora de imitação durante o PPO.** Este é o ponto central do DAPG
   (Rajeswaran et al., 2018): BC como **pré-treino apenas não basta** — o método que
   funciona mantém um termo de BC/KL **durante** o RL, com peso decaindo
   ($w(s,a)=\lambda_0\lambda_1^k \max A$). A revisão da literatura de fine-tuning
   ("Fine-tuning RL models is secretly a forgetting mitigation problem",
   arXiv:2402.02868) classifica os antídotos que funcionam: loss BC auxiliar sobre
   buffer de demos, KL para a política pré-treinada (*kickstarting*), EWC. Warm-start
   "seco" (só pesos, sem âncora) é justamente a configuração que essa literatura prevê
   que esquece.

**Previsão falseável de H1:** ator BC avaliado em rollout (sem PPO) tem grasp ≫ 0, e o
grasp da iteração ~0 do PPO (com ruído/normalização da configuração que falhou) é ≈ 0.

### H2 — Aliasing parcial real no frame único (observabilidade) — suporte moderado, provavelmente secundária

Há variáveis latentes das quais a ação ótima depende e que **não** estão num frame:

| Variável latente | Num frame único? | Por que importa |
|---|---|---|
| Tempo de contato até a interceptação | ✅ derivável (pos. rel. + vel. peça, se velocidade da esteira constante) | timing da descida/fecho |
| Velocidade da esteira (sorteada por episódio) | ⚠️ só via vel. da peça enquanto ela desliza livre | após contato/captura, some |
| **Estado de captura** (peça travada à garra × repouso sobre a esteira) | ❌ vel.≈0 e pos. rel. parecidas nos dois casos; força instantânea ajuda mas é transitória | decide "levantar" × "continuar fechando/esperar" |
| **Fase do episódio / tentativa** (aproximação × nova tentativa após grasp falho) | ❌ estados geometricamente iguais pedem ações opostas (insistir × abortar p/ place) | aliasing clássico de POMDP |
| Histórico de força de contato (pico transitório entre passos de controle) | ❌ força amostrada pode perder o evento | distingue "tocou" de "segurando" |
| Escorregamento/atrito durante o lift | ❌ só aparece integrado no tempo | segurar × re-fechar |

A teoria é inequívoca: em POMDP, a política ótima é função da **história** (belief
state), não do frame (DRQN, Hausknecht & Stone 2015; Heess et al. 2015). A prática de
manipulação com contato segue isso: trabalhos atuais de grasping sob oclusão parcial
usam GRU/LSTM justamente para inferir dinâmica da peça a partir da história
(ex.: graspability-field, arXiv:2606.30474). Frame stacking cobre janelas curtas
(< ~1 s); recorrência cobre dependência de eventos a segundos de distância
(síntese em embodiedbook §32.6).

**Mas atenção ao escopo:** o aliasing acima afeta sobretudo *gerenciar* o grasp
(levantar na hora certa, recuperar de falha). A *iniciação* geométrica do grasp é
majoritariamente identificável no frame. E o contra-teste é forte: o ator BC é
igualmente feedforward e sem memória — se ele, em rollout, pega a peça, aliasing não
é o gargalo do grasp. **H2 não explica grasp 0,0 desde o início; H1 explica.**

### H3 — Crítico simétrico fraco (sinal de vantagem pobre) — suporte como co-fator

Ator e crítico veem as mesmas 24 dims parciais → o crítico não consegue avaliar bem
estados aliasados → vantagens ruidosas → aprendizado lento/instável. Crítico
privilegiado/assimétrico é prática padrão e barata (§4), mas **não** devolve ao ator
informação que ele não tem na decisão, nem protege o warm-start. Co-fator, não causa
raiz provável do zero absoluto.

### H4 — BC não transfere nem em rollout (covariate shift / aliasing no BC) — a testar, não assumir

BC treina na distribuição do professor; em rollout a política visita estados fora
dela e erros compõem (Ross et al. 2011 — DAgger). Se o ator BC puro já der grasp ≈ 0
em rollout, o problema está **antes** do PPO: ajuste do BC, aliasing (H2) ou ambos.
Esta hipótese só pode ser separada de H1 medindo — é o Estágio A do Exp. 1.

---

## 2. Q1 — O grasp é identificável num frame único? O que diz a literatura

- **Iniciação do grasp: sim, em grande parte.** Posição relativa + velocidade da peça
  + propriocepção + força instantânea determinam onde/quando interceptar, sob
  velocidade de esteira constante por episódio.
- **Gestão do grasp e recuperação: não.** Captura, fase/tentativa e eventos de contato
  transitórios são latentes (§1, H2). O professor — que enxerga o estado completo —
  precisou de uma **máquina de fases com latch de captura** para resolver a tarefa;
  isso é evidência interna forte de que a estrutura temporal importa.
- **Frame stacking × RNN:** stacking é o remédio barato quando a fase é recuperável
  de <1 s de movimento; RNN (LSTM/GRU) quando a decisão depende de eventos segundos
  atrás (ex.: "já tentei e falhei"), ao custo de treino sequencial mais delicado
  (armazenamento por trajetória, reset de hidden state). Isaac Lab suporta história
  nativamente (grupo de observação com `history_length`) e o RSL-RL suporta RNN (§3) —
  os dois degraus estão disponíveis sem trocar de stack.

## 3. Q2 — RSL-RL atual suporta política recorrente com Isaac Lab 3.0? Sim

Duas gerações de API coexistem na literatura; a que vale para esta stack é a nova:

- **RSL-RL atual (≥3, tensordict):** modelos `MLPModel`/`CNNModel`/`RNNModel`; o PPO
  tem gerador de mini-batches recorrente que preserva trajetórias, hidden states por
  ambiente e máscaras de reset (documentado no DeepWiki do projeto).
- **Wrapper Isaac Lab (`isaaclab_rl.rsl_rl`):** existe `RslRlRNNModelCfg`
  (subclasse de `RslRlMLPModelCfg`) com `class_name="RNNModel"` (default),
  `rnn_type ∈ {"lstm","gru"}`, `rnn_hidden_dim`, `rnn_num_layers`, mais `hidden_dims`,
  `activation` e `obs_normalization` do MLP. Configuração = apontar **actor** (e, se
  quiser, critic) para `RslRlRNNModelCfg(...)` na cfg do agente; como o Isaac Lab 3.0
  converte a cfg legada via `handle_deprecated_rsl_rl_cfg` para dicts actor/critic com
  `class_name`, o caminho é o mesmo já usado no projeto para o MLP.
- **Geração antiga (RSL-RL 2.x):** `class_name="ActorCriticRecurrent"` com
  `rnn_type`, `rnn_hidden_size`, `rnn_num_layers` na policy cfg — aparece em muito
  material de Isaac Lab ≤2.x; **não** confundir os campos (`rnn_hidden_size` ×
  `rnn_hidden_dim`).
- **Rota professor-aluno:** o ecossistema também tem `StudentTeacherRecurrent` +
  algoritmo `Distillation`, caso se opte por destilar o professor em aluno recorrente.
- **Custos/ressalvas:** treino mais lento por iteração e sensível a
  `num_steps_per_env` (sequências); playback precisa carregar hidden state entre
  passos (o runner trata); BC para ator recorrente exige treino **por sequências**
  (não pares i.i.d.) — por isso RNN não é o primeiro experimento.

## 4. Q3 — Crítico privilegiado / assimétrico: evidência

- Origem: Pinto et al. 2017 (asymmetric actor-critic) — crítico com estado completo
  reduz variância e melhora as vantagens; o ator mantém obs realizáveis.
- Prática consolidada no Isaac Lab: grupo de observação `critic` separado do
  `policy`, roteado via `obs_groups = {"actor": ["policy"], "critic": ["policy",
  "critic"]}`; é o arranjo por trás das políticas de locomoção deployadas (ANYmal:
  crítico com height-scan privilegiado, ator só com propriocepção).
- Ablação recente útil (RAFT, 2026): ator feedforward **sem memória** + crítico
  privilegiado fechou 79% do gap para o oráculo — o privilégio do crítico foi o
  mecanismo primário; recorrência adicionou o restante. Lição: assimetria compra
  muito sinal de treino barato, mas **não** resolve aliasing do ator.
- **Nesta tarefa**, o grupo privilegiado natural do crítico: velocidade verdadeira
  da esteira, estado de captura, velocidade da peça no referencial do mundo,
  fase do professor (no crítico, fase é legítima sem discussão — ele é descartado
  no deploy), massa/atrito da peça.

## 5. Q4 — Features de fase/estado como observação do ator: legítimo?

**Regra:** é legítimo se a feature é **estado do mundo/tarefa computável no
deploy** (pelo controlador da célula, sensores ou relógio), e vazamento se é
**estado interno da decisão do professor** (plano, alvo de IK escolhido, intenção).

- **Precedente forte:** variáveis de fase/clock (sin/cos de φ) são observação padrão
  em locomoção humanoide/quadrúpede e em tracking loco-manipulativo — ninguém trata
  como vazamento; é o relógio da tarefa.
- **Aplicação aqui — dar os ingredientes, não o rótulo do professor:**
  - ✅ tempo decorrido do episódio (normalizado) — relógio da célula;
  - ✅ velocidade comandada/medida da esteira — o drive da esteira conhece;
  - ✅ flag "peça assegurada" **derivada de sensores** (garra fechada + força
    bilateral sustentada) — um gripper real com sensor de força/posição reporta isso;
  - ⚠️ índice da fase do professor (approach/grasp/lift/place) — é uma discretização
    de estado físico, defensável em simulação, mas cria ficção de deploy se a célula
    real não computar a mesma máquina; preferir os ingredientes acima, que a
    política pode combinar sozinha;
  - ❌ ponto de interceptação calculado pelo IK do professor, sua fase *futura*,
    qualquer coisa que só exista dentro do script do professor.

## 6. Q5 — Como a literatura trata o colapso BC→PPO (e os antídotos, por evidência)

1. **DAPG (Rajeswaran et al. 2018) — evidência mais forte e diretamente aplicável:**
   BC no pré-treino **e** termo de imitação no gradiente durante o RL, com peso
   decaindo por iteração. Em manipulação destreza com recompensa esparsa, supera
   largamente BC→RL "seco". Observação dos revisores: BC+NPG (mantendo o termo)
   ≈ DAPG em 3/4 tarefas — o ingrediente ativo é **não largar a imitação durante
   o RL**.
2. **Kickstarting / KL para a política congelada do BC** (Schmitt et al. 2018;
   revisado em arXiv:2402.02868): penalidade KL(π_BC ‖ π) sobre estados visitados
   online, decaindo. Primo próximo do DAPG, fácil de implementar como loss auxiliar.
3. **Higiene de fine-tuning (FLaRe, arXiv:2409.16578):** remover bônus de entropia
   no início, não compartilhar features ator-crítico, tratar o crítico com cuidado —
   ablações mostram colapso ao remover componentes. Barato de adotar já.
4. **Crítico aquecido / ator congelado no início:** mitiga vantagens-lixo dos
   primeiros updates (evidência indireta da literatura de inicialização de Q em
   fine-tuning, arXiv:2505.16856: crítico incompatível ⇒ queda inicial).
5. **Congelamento parcial / EWC / LR menor no ator:** funcionam como freio de
   esquecimento, mas são o antídoto mais fraco/mais cego dos cinco — úteis como
   complemento, não como estratégia.
6. **DAgger:** se o BC puro falhar em rollout (H4), o remédio canônico é rotular
   estados visitados pela política com o professor — aqui é **viável de verdade**,
   porque o professor é scriptado e roda dentro do próprio simulador.

---

## 7. Experimentos candidatos (ranqueados por custo × poder decisivo)

### Exp. 1 — Diagnóstico em 2 estágios + warm-start protegido **[recomendado]**
Um único kernel: (A) avalia o ator BC puro em rollout; (B) roda o PPO warm-start com
o pacote de antídotos de custo ~zero (ruído inicial baixo, entropia 0, normalização
consistente, crítico aquecido). **Poder decisivo máximo:** bissecta H1 × H4 e já
testa a correção de H1 na mesma corrida. Custo: ≈ o da corrida BC v7 (o Estágio A
adiciona minutos). Protocolo completo na §8.

### Exp. 2 — Estado observável ampliado (features + crítico assimétrico), depois memória
(2a) Adicionar ao ator: tempo normalizado do episódio, velocidade da esteira, flag
"peça assegurada" derivada de sensores, **ação anterior**; crítico ganha grupo
privilegiado (§4). MLP igual, BC reaproveitado (features são função do estado já
gravado nas demos — exige re-coletar ou derivar das demos existentes).
(2b) Se 2a não bastar: frame stacking (história curta via grupo de observação) e/ou
ator `RNNModel` (LSTM 128–256), com BC por sequências.
Custo médio (mudança de obs invalida checkpoints e o BC atual); poder decisivo alto
para H2/H3, mas **cego para H1** se rodado antes do Exp. 1 — por isso é o segundo.

### Exp. 3 — DAgger em simulação / destilação professor→aluno recorrente
Professor scriptado rotula estados visitados pela política (DAgger) ou treino por
`Distillation` com aluno `StudentTeacherRecurrent`. Ataca H4 e H2 juntos, mas é a
maior obra de engenharia das três e só se justifica com o diagnóstico do Exp. 1
apontando BC/rollout ou aliasing. Custo alto, poder decisivo alto, **depois**.

---

## 8. Protocolo pré-registrado — Experimento nº 1 (Kaggle)

**Nome:** OBS-1 — "O BC pega? O PPO preserva?" (kernel novo, branch da mesma base da
corrida BC v7; nada de treino na VM.)

**Constantes herdadas (controle):** mesma tarefa/reward/currículo; coleta do
professor idêntica (portão: sucesso do professor ≥ 0,80; referência 0,865);
BC idêntico (300 épocas, mesma arquitetura [256,128,64], mesmas demos/seeds);
PPO: 256 envs × 1500 iterações, budget de amostras igual ao da v7 (~9,2M).

**Estágio A — BC puro em rollout (antes de qualquer PPO, no mesmo kernel):**
- Carregar `bc-actor` e executar **rollout determinístico** (ação = média, sem ruído
  de exploração, sem atualização) por **200 episódios** nas condições/seeds de
  avaliação padrão do projeto.
- Registrar: `grasp_rate_BC`, `success_rate_BC`, e os estágios atingidos
  (contato/fecho/lift/place) por episódio.
- Leitura pré-registrada: `grasp_rate_BC ≥ 0,30` ⇒ material BC viável (H4 fraca);
  `grasp_rate_BC < 0,05` ⇒ BC não transfere nem sozinho (H4/H2 sobem; ver MATA-B).

**Estágio B — PPO warm-start protegido (únicas diferenças vs. a corrida que zerou):**
1. `init_noise_std = 0,15` (ação amostrada perto da média clonada);
2. `entropy_coef = 0,0`;
3. normalização de observação **idêntica** à do BC (obs brutas nos dois; verificar
   por construção — mesmo pipeline de obs, `obs_normalization=False`, sem
   estatísticas empíricas novas sobre o ator);
4. **crítico aquecido:** ator congelado por 50 iterações (só o crítico atualiza),
   depois treino normal;
5. log obrigatório: `grasp_rate` das **5 primeiras iterações** (iteração-0 = retrato
   do warm-start) e curva de grasp/success a cada 50 iterações.
(Se o runner não permitir congelar o ator por configuração, implementar via LR do
ator = 0 nas primeiras 50 iterações no script de treino — registrar o mecanismo
usado no relatório da corrida.)

**Métricas oficiais (fim da corrida):** benchmark padrão do projeto →
`success_rate` (primária), `grasp_rate` final e máximo, grasp@iteração-0,
`grasp_rate_BC`/`success_rate_BC` do Estágio A, loss final do BC, sucesso do
professor na coleta. O notebook imprime o veredito (protocolo da casa).

**Portões (contra o 0,0 atual — PPO puro e BC v7):**
- **PROVA:** benchmark `success_rate ≥ 0,30` ⇒ a rota warm-start está validada e a
  causa era a interface (H1); próximos passos viram escala/robustez.
- **PARCIAL:** qualquer um de —
  (i) `0 < success_rate < 0,30`;
  (ii) `success_rate = 0` mas `grasp_rate` final ≥ 0,20 (grasp recuperado; o gargalo
  migrou para lift/place — problema localizado, não mais "não pega");
  (iii) grasp alto nas primeiras ~100 iterações e decaindo depois (antídoto
  parcial ⇒ próximo experimento = termo DAPG/KL durante o PPO).
- **MATA:** `success_rate = 0` **e** `grasp_rate` final ≤ 0,02 — com duas leituras
  obrigatórias antes de encerrar:
  - **MATA-A:** se `grasp@iteração-0 ≥ 0,20` e mesmo assim colapsa ⇒ H1 na forma
    "esquecimento durante o PPO" confirmada por eliminação; Exp. seguinte = DAPG
    (termo de imitação no loss), **não** memória.
  - **MATA-B:** se `grasp_rate_BC < 0,05` já no Estágio A ⇒ o veredito é sobre o
    **BC**, não sobre o PPO: a frente muda para Exp. 2 (estado ampliado/memória)
    e/ou DAgger (Exp. 3); declarar a corrida como "diagnóstico concluído", não
    como fracasso do warm-start.

**Integridade:** seeds e protocolo de avaliação idênticos aos do benchmark S68/v7;
qualquer desvio de configuração além dos 4 itens do Estágio B invalida a corrida
como teste de H1 e deve ser registrado como tal no relatório.

---

## Referências principais

- Rajeswaran et al. 2018 — *Learning Complex Dexterous Manipulation with Deep RL
  and Demonstrations* (DAPG). https://arxiv.org/pdf/2301.07302 (apêndice A.1) e
  revisão em https://ar5iv.labs.arxiv.org/html/1910.04281
- Wolczyk et al. 2024 — *Fine-tuning RL Models is Secretly a Forgetting Mitigation
  Problem*. https://arxiv.org/pdf/2402.02868
- FLaRe — *Achieving Masterful and Adaptive Robot Policies with Large-Scale RL
  Fine-Tuning* (entropia/feature-sharing no fine-tuning). https://arxiv.org/html/2409.16578v2
- Inicialização de crítico em fine-tuning online. https://arxiv.org/html/2505.16856v1
- Isaac Lab — API `isaaclab_rl.rsl_rl` (`RslRlRNNModelCfg`, `RslRlMLPModelCfg`,
  `obs_normalization`). https://isaac-sim.github.io/IsaacLab/main/source/api/lab_rl/isaaclab_rl.html
- RSL-RL — arquitetura (MLPModel/RNNModel, mini-batches recorrentes).
  https://deepwiki.com/leggedrobotics/rsl_rl/3-core-architecture
- Crítico assimétrico no Isaac Lab (grupos policy/critic, ANYmal).
  https://embodiedbook.icsgen-ai.org/part-4-reinforcement-learning-for-embodied-agents/module-17-massively-parallel-and-gpu-rl/section-17.3.html
- RAFT — crítico privilegiado como mecanismo primário (ablação).
  http://arxiv.cnoctave.top/pdf/2608.22976
- Fase/clock como observação — locomoção humanoide (sin/cos de φ).
  https://arxiv.org/html/2504.13619v1 ; visão baseada em locomoção bípede (clock
  por perna). https://arxiv.org/html/2309.14594v2
- Memória em grasping sob observabilidade parcial (GRU no aluno).
  https://arxiv.org/pdf/2606.30474
- Frame stacking × recorrência (janelas de decisão). https://embodiedbook.icsgen-ai.org/part-7-language-vision-and-action/module-32-vision-language-models-for-embodiment/section-32.6.html
- DRQN / POMDP clássico. https://sugaku.net/oa/W2987246610/
