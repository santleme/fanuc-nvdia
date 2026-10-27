# FANUC conveyor pick-and-place

Projeto spec-driven para construir, validar e executar no Kaggle uma célula de
manipulação industrial inspirada no FANUC M-10iD/12. O cenário tem esteira,
peça, garra, ponto de retirada e zona de depósito.

## Estado (27/10/2026)

A pipeline completa está verde desde a v67: probe, baseline físico,
treino PPO, playback com vídeo e benchmark oficial rodando de ponta a
ponta no Kaggle (conta `pascalcoimbra`, T4). Marcos recentes:

- **v66/v67** — destravada a física OvPhysX (diagnóstico de clone modes:
  a junta `--fix-base` do conversor envenenava os clones; correção =
  `fix_root_link` no spawn, veredito na SPEC). Primeiro treino completo
  e, com 64 envs, treino de 421 s; **success_rate oficial 0,0** — a
  infraestrutura fechou e o gargalo virou aprendizado.
- **v68 (rodando)** — reward em estágios + currículo de esteira +
  observações de contato, 256 envs × 1500 iterações (issue #1;
  protocolo e portões PROVA/PARCIAL/MATA na SPEC).
- **Spike Newton/MJWarp (rodando)** — mesmo pipeline com
  `physics=newton`, em kernel próprio `fanuc-newton-spike` (issue #2).
- **Na fila** — BC warm-start a partir do professor scriptado (issue #3,
  aberta), condicionado ao sinal da v68.

Decisões e evidências: [SPEC.md](SPEC.md) (protocolos e vereditos),
`docs/integracoes-decisao-2026-10-27.md` (integrações avaliadas),
`docs/audit-2026-10-26-v65.md` (auditoria profunda) e
`outputs-v67/` (artefatos da primeira pipeline verde, incl. o vídeo da
política). Backlog pós-v68 em `docs/backlog-pos-v68.md`.

## Modelo de referência

FANUC M-10iD/12: 6 eixos, carga nominal de 12 kg, alcance máximo de 1.441 mm,
repetibilidade publicada de ±0,02 mm. Esses dados dimensionam o cenário, mas não
substituem geometria articulada, inércias, controladores ou dados de calibração.

Fonte: [FANUC America — M-10iD/12](https://www.fanucamerica.com/products/robot/m-10id-12).

## Rodar o baseline local

```bash
python3 -m pip install -e '.[dev,video]'
PYTHONPATH=src pytest -q
PYTHONPATH=src python3 scripts/render_demo.py --output outputs/fanuc-cell-demo.mp4
```

## Envio e execução no Kaggle

A CLI local está integrada para `pascalcoimbra`. Gere o notebook a partir do
código e envie/executa uma nova revisão:

```bash
python3 scripts/build_kaggle_notebook.py --kaggle-user pascalcoimbra
kaggle kernels push -p kaggle
```

Confira com `kaggle kernels status pascalcoimbra/fanuc-conveyor-pick-place`
e leia os logs com `kaggle kernels logs pascalcoimbra/fanuc-conveyor-pick-place`.
O notebook detecta hardware/driver, roda os testes do pacote e mede conversão,
baseline físico, treino e playback separadamente. Para reduzir tempo de
depuração, não repete o smoke Cartpole nem renderiza o vídeo cinemático em cada
execução; o baseline só inicia treino e vídeo físico depois de confirmar
grasp/place sob OvPhysX.

Fonte: [anúncio Kaggle sobre aposentadoria da P100 e T4x2](https://www.kaggle.com/product-announcements/735239).
