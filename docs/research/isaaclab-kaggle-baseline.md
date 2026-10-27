# Isaac Lab kit-less no Kaggle: decisão de runtime

**Atualizado:** 2026-10-25. O alvo deste projeto é Isaac Lab 3.0.0 com
OvPhysX, usando URDF para descrever o braço. A célula inclui esteira, peça,
pedestal, garra de dois dedos, sensores de contato, baseline determinístico,
treino PPO e gravação de vídeo. O asset é um proxy inspirado no M-10iD/12, não
CAD ou modelo oficial da FANUC.

## Decisão técnica

- OvPhysX executa PhysX sem Omniverse Kit. Isaac Lab 3.0 lista articulações,
  objetos rígidos e sensores de contato agregados/filtrados como componentes
  suportados. O ambiente local não pode confirmar esses recursos em runtime;
  o smoke e o teste de agarramento no Kaggle permanecem necessários.
- O projeto usa `OvPhysxCfg` e adiciona `physics=ovphysx` em smoke, baseline,
  treino e avaliação. O projeto externo gerado pelo Isaac Lab suporta o extra
  `ovphysx` com `uv run --extra ovphysx ...`.
- A conversão é URDF → USD pela implementação oficial `convert_urdf.py` e o
  extra standalone `importers`. Na tag v3.0.0, o utilitário oficial escolhe
  Newton/MJWarp internamente quando o importador standalone está ativo. Para
  manter todo o processo em OvPhysX, o notebook copia esse utilitário para um
  arquivo temporário e troca somente o backend padrão por `ovphysx`; a cópia
  aborta caso a linha upstream não corresponda ao código revisado. Assim a
  conversão não inicia solver MuJoCo-Warp nem depende de Isaac Sim/Kit. O comando
  resultante segue o mesmo utilitário e argumentos oficiais:

  ```bash
  uv run --extra importers --extra ovphysx isaaclab -p \
    scripts/tools/convert_urdf_ovphysx.py \
    assets/m10id12_robot.urdf /path/to/assets --fix-base \
    --joint-stiffness 0.0 --joint-damping 0.0 --joint-target-type none
  ```

  Conversão não valida por si só massa, inércia, colisões, limites, atuadores ou
  resultado da tarefa. O notebook inspeciona o USD e exige seis juntas
  revolutas, dois deslizantes e corpo `tool0` antes de instanciar a tarefa.
- Para MP4 no processo sem Kit, o notebook seleciona explicitamente `newton_gl`
  com modo headless e o extra de vídeo.
  A documentação do Isaac Lab 3.0 lista esse visualizador como
  compatível com OvPhysX. Aqui Newton GL é somente o visualizador; o solver da
  simulação continua sendo OvPhysX.
- O caminho precisa de GPU NVIDIA CUDA. A TPU Kaggle não substitui esse runtime.
  O notebook identifica GPU, arquitetura e driver antes de instalar Isaac Lab.
  O perfil configurado é T4, internet ligada e TPU desligada; a sessão Kaggle
  real precisa confirmar o dispositivo e as versões visíveis.

Fontes primárias: [OvPhysX no Isaac Lab 3.0](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/concepts/physics-backends/ovphysx.html),
[extras e projetos externos](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/source/developer-tools/template_generator.html),
[conversão de URDF](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/how-to/import_new_asset.html),
[visualizadores e vídeo](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/concepts/renderers.html),
[seletores de backend](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/concepts/backends_and_presets.html).

## Estado do envio Kaggle

`kaggle` 2.2.4 está instalado e a configuração local existe; nenhum valor de
credencial foi aberto. `kaggle kernels push -p kaggle` falha antes de autenticar,
na resolução DNS de `api.kaggle.com`. O ambiente declara
`CODEX_SANDBOX_NETWORK_DISABLED=1`; as chamadas de rede também falham nesse
sandbox. Assim, o notebook e a configuração podem ser validados localmente, mas
publicação, execução, vídeo remoto, runtime OvPhysX e métricas PPO ainda não têm
evidência Kaggle.
