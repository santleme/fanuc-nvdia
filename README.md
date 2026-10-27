# FANUC conveyor pick-and-place

Projeto spec-driven para construir, validar e executar no Kaggle uma célula de
manipulação industrial inspirada no FANUC M-10iD/12. O cenário tem esteira,
peça, garra, ponto de retirada e zona de depósito.

## Estado

O spec está registrado em [SPEC.md](SPEC.md). O projeto inclui um proxy
articulado de seis eixos, IK posicional e de pose completa, sequência de
interceptação/place, URDF, e uma tarefa Isaac Lab manager-based com PPO,
sensores de contato bilaterais e critérios de sucesso/falha.

O notebook Kaggle em `kaggle/fanuc-conveyor-pick-place.ipynb` é enviado e
executado com a CLI integrada da conta `pascalcoimbra`. A revisão 28 detectou
2× Tesla T4, inicializou Isaac Lab 3.0.0/OvPhysX, passou os smokes oficial e da
tarefa customizada, mas falhou no grasp físico. As revisões 29/30 descobriram
o erro de tempo da esteira, deflexão no TCP e a peça presa à esteira durante o
lift. A versão 36 registrou 77 passos com contato bilateral fechado, mas não
levantou a peça. A análise do URDF encontrou um provável excesso de fechamento:
25 mm por dedo atravessa a largura de 80 mm; o contato esperado é perto de 6 mm.
A versão 37 foi salva no Kaggle sem execução com fechamento parcial, captura
compatível e diagnóstico da pausa da esteira. A CLI ainda lista versões 35/36
como RUNNING apesar dos logs mostrarem falha terminal, e recusou novo run pelo
limite de duas sessões. O CLI oficial não tem comando de cancelamento; o SDK
retornou 403. É preciso liberar essas duas sessões na interface Kaggle e então
executar a 37, validando grasp/lift/place antes de liberar PPO, vídeo físico e
avaliação. A v42 mediu 81,9 s no `uv sync` do Isaac Lab, 50,5 s no smoke
Cartpole, 57,2 s na sincronização do projeto customizado e 21,1 s no setup local
e render cinemático. O pedido explícito de L4 ainda recebeu 2×T4, então voltei a
deixar T4 como acelerador padrão. A v43 está salva sem execução: não instala
extras já presentes, não renderiza o vídeo cinemático em cada tentativa e roda o
baseline antes de instalar o ambiente PPO ou executar smokes redundantes. O
alvo foi ajustado para 14 mm após 10 mm produzir só ~2 N por dedo. O Kaggle
recusou executar v43 porque ainda conta v40/v42 como sessões ativas, apesar dos
logs terminais.

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
