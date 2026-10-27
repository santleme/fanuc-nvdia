# Decisão de integrações — FANUC × NVIDIA (2026-10-27)

Decisão pedida pelo fundador ("analisa os projetos, decide se faz sentido
integrar, sobe issues e resolve"): pesquisa em três frentes sobre os
projetos candidatos mapeados para este repo, confrontada com o estado
real do projeto pós-v67 (pipeline verde, success_rate 0.0, gargalo =
aprendizado). Este documento registra os vereditos; o trabalho vira
issues no repo e a primeira fatia entra por PR.

## O diagnóstico que manda na decisão

A v67 não falhou por hiperparâmetro nem por engine: o MDP torna o
evento recompensado estatisticamente inalcançável por exploração
aleatória. Causas raiz (detalhe na issue #1):

- o termo `grasp` é uma conjunção de 4 condições paga uma vez, sem
  degraus intermediários (nada paga "tocou", "fechou sobre a peça");
- `lift` exige a peça já levantada — sem grasp não há sinal;
- `place` NÃO é condicionado ao grasp: a peça parada na esteira rende
  ~0,05–0,1/passo de graça → ótimo local (reward 0,738 sem tocar na
  peça), exatamente o comportamento observado;
- o `reach` mira o CENTRO da peça com kernel estreito, empurrando o
  TCP para dentro da peça (a pose de grasp é deslocada de
  `(0,095; 0; −0,02)` — o `tool_to_part_offset` do baseline);
- peça móvel em passe único, dificuldade cheia desde a iteração 0,
  sem currículo; sensores de contato fora das observações (a política
  fecha a garra "no escuro"); orçamento de 768 mil amostras para um
  problema que a tarefa oficial análoga roda com ~100× mais.

## Vereditos por projeto

| Projeto | Veredito | Porquê |
|---|---|---|
| **Newton (MuJoCo-Warp) no Isaac Lab 3.0** | **INTEGRAR — spike agora** | Backend oficial e GA no Isaac Lab 3.0.0 (é a dependência de física padrão da versão); sensor de contato com paridade de API confirmada no código — expõe `normal_force_matrix_w`, o campo exato que o MDP lê. Troca = 1 campo de preset + token de CLI, sem tocar na tarefa. Comparação independente no mesmo Isaac: stepping 1,93–2,23× mais rápido, 5,3× menos VRAM. Responde se o contato macio da família MuJoCo destrava o grasp — sem pagar migração e sem tocar na camada de vídeo (critério do fundador). |
| **ManiSkill** | **REFERÊNCIA** | `PickCube-v1` é o gabarito público de reward em estágios (aproximar → agarrar → levantar → levar) e de demos por motion planning. Como plataforma é PhysX-por-PhysX: movimento lateral, perde USD/Newton/RTX. |
| **Reward staged + currículo (padrão das tarefas oficiais do Isaac Lab)** | **INTEGRAR — é a v68** | Ataca todas as causas raiz acima dentro do próprio MDP: reach no ponto de grasp com kernel largo, degraus de grasp, lift binário por passo, place condicionado ao grasp, observações de contato, currículo de velocidade da esteira e escala 256 envs × 1500 it. Protocolo pré-registrado na SPEC. |
| **Demonstrações do professor scriptado → BC warm-start** | **INTEGRAR — segunda onda** | O projeto já tem o professor: o baseline determinístico passa no grasp desde a v51 e roda no próprio env com a IK DLS da casa. Registrar (obs, ação) e pré-treinar o ator por BC é barato; entra se a v68 mostrar grasp mas success baixo (ou direto, se a v68 der MATA). |
| **cuRobo / SkillGen / LeRobot** | **NÃO integrar neste ciclo** | cuRobo: integração pesada e frágil no stack kit-less (fonte + CUDA toolkit + commit pinado), licença V1 não-comercial, ganho ~0 sobre a IK da casa para demos por waypoints. LeRobot: formato para dataset público de teleop com câmeras; aqui as demos são vetores de estado — `.pt`/`.npz` basta. LeRobot volta se houver plano de publicar dataset. |
| **Genesis** | **DESCARTAR** | Velocidade de marketing não se sustenta em cena rica em contato (benchmark independente: 3–10× mais lento que Isaac/ManiSkill com colisões; câmera derruba a velocidade); sem baseline de pick-and-place para calibrar reward; render fora do ecossistema USD/RTX rebaixa o teto de vídeo do fundador. Não resolve nem a dor histórica (contato) nem a atual (reward). |
| **MuJoCo/MJX puro** | **REFERÊNCIA — migração segue engatilhada** | Nenhum gatilho da auditoria de 26/10 disparou. Fatos novos reforçam o híbrido: Newton 1.0 GA, Isaac Sim 6.2 já embarca Newton 1.6.1 + MuJoCo 3.12, MJX aceita MuJoCo-Warp como backend. A física da família MuJoCo se obtém por preset dentro do Isaac — a migração pura (4–7 dias + retuning) compra ainda menos que ontem. |
| **robosuite** | **REFERÊNCIA** | Evidência publicada de que ação em espaço operacional (OSC) aprende mais rápido que posição de junta — insumo para uma futura troca de espaço de ação, se o gargalo migrar para lá. CPU-only, fora como plataforma. |
| **Eureka (reward por LLM)** | **ADIAR** | 48–80 treinos por tarefa no laço evolutivo (12–20 h de GPU) para otimizar um MDP que primeiro precisa ser consertado à mão. Versão de bolso (4–6 variantes de pesos em screenings curtos) fica como reserva pós-v68. |

## Ordem de execução

1. **v68 — reward staged + currículo + escala** (issue #1): o gargalo.
2. **Spike Newton/MJWarp** (issue nova): barato, reversível, portão
   próprio comparável à v67; roda em kernel separado.
3. **BC warm-start do professor** (issue nova): condicionado ao sinal
   da v68 (grasp > 0 com success baixo → entra; grasp ≈ 0 na fase
   estática → vira o experimento principal).

Protocolos completos (hipótese, baseline, portões PROVA/PARCIAL/MATA)
na SPEC e nas issues. Regra da casa mantida: código e testes na VM,
experimento só no Kaggle, veredito impresso pelo notebook.

## Fontes

- Isaac Lab 3.0 — release notes (multi-backend, `isaaclab_newton`,
  sensores de contato): https://isaac-sim.github.io/IsaacLab/main/source/refs/release_notes.html
- Newton (GitHub, 27/10/2026: 5.846 stars, v1.7.0 em 14/10/2026,
  Linux Foundation — NVIDIA/DeepMind/Disney): https://github.com/newton-physics/newton
- Comparação independente PhysX × Newton no Isaac Lab 3.0.0:
  https://yusufdxb.github.io/papers/physx-newton-backend-comparison.pdf
- Tarefa oficial Lift do Isaac Lab (reward staged, lift binário,
  goal tracking condicionado): https://github.com/isaac-sim/IsaacLab
- ManiSkill / PickCube-v1: https://github.com/mani-skill/ManiSkill
- Benchmark independente do Genesis: https://stoneztao.substack.com/p/the-new-hyped-genesis-simulator-is
- Eureka: https://arxiv.org/html/2310.12931v2
- Auditoria interna Isaac × MuJoCo (26/10/2026):
  `docs/audit-isaac-vs-mujoco-2026-10-26.md`
