# Backlog pós-v68 — issues prontas para abrir (27/10/2026)

Levantadas na sessão noturna de 27/10/2026 a partir das evidências das
v65–v67 e da auditoria, deduplicadas contra as issues existentes
(#1 reward/currículo — resolvida na v68; #2 Newton — spike rodando;
#3 BC warm-start — aberta). Abertura formal no GitHub pendente de
aprovação do fundador (o conector exige clique dele); os corpos abaixo
estão prontos para virar issue como estão.

## B1 — Diagnóstico de modos de perda (object_lost) [contingência] — ABSORVIDA pela issue #8 (observabilidade/memória), aberta em 27/10/2026
Se a v68 fechar PARCIAL ou MATA, rodar kernel instrumentado com a
melhor política disponível classificando cada episódio perdido:
transitou a esteira sem interceptação × derrubada pelo braço × caiu
após grasp × timeout. Sem esse mapa, redesign de reward é chute.
Aceitação: relatório com a distribuição dos modos em ≥64 episódios e
recomendação de qual termo/currículo atacar.

## B2 — Modo eval-only no notebook (sem retreino)
Hoje toda avaliação repete install+treino (~15 min). Empacotar o
checkpoint como dataset Kaggle e adicionar modo que pula direto para
playback+benchmark a partir dele. Aceitação: avaliar um checkpoint
existente em <8 min de kernel, com success_rate idêntico ao da corrida
original no mesmo seed.

## B3 — Benchmark multi-seed no protocolo padrão — IMPLEMENTADA em 27/10/2026 (issue #10; builder com seeds 101/202/303, média±desvio no runtime-report, SPEC atualizada)
success_rate de seed único não separa sinal de ruído. Adotar 3 seeds
fixas no benchmark oficial e reportar média ± desvio em todo veredito.
Aceitação: SPEC atualizada e próximo veredito já reporta os 3 seeds.

## B4 — Cache da instalação do Isaac Lab em dataset
Desenho registrado na SPEC (veredito v67): kernel construtor gera o
ambiente instalado uma vez, kernels seguintes montam o dataset em vez
de clone+sync (~4–6 min/corrida, payback ~6 corridas, com fallback para
install completo se a importação falhar). Aceitação: corrida completa
com o cache, tempo de setup medido contra a linha de base.

## B5 — Probe+baseline num único boot do Isaac Sim [spike de infra]
A fusão foi adiada na v67 (economia <1 min × risco de reentrada do
launch_simulation não provada). Spike isolado com portão próprio:
se os dois estágios rodarem no mesmo processo com os mesmos
vereditos, adota; se não, encerra sem custo. Aceitação: veredito do
spike registrado na SPEC.

Implementação pronta (issue #14, 27/10/2026): entrypoint
`scripts/probe_and_baseline.py` + flag `--single-boot-spike` no
builder; protocolo pré-registrado na SPEC. Veredito aguarda a
corrida do spike no Kaggle.

## B6 — Banda ambígua do limiar da garra (0,004–0,005 m)
Achado da auditoria de 26/10: o limiar de fechamento tem uma banda sem
comportamento definido entre o valor do código e o dos testes antigos.
Probe dedicado varrendo o limiar contra força de contato medida.
Aceitação: valor único justificado por medição, código e teste
alinhados.

## B7 — CI no GitHub Actions (núcleo puro) — ENTREGUE em 27/10/2026 (commit 57c83c9; 49/49 em venv limpo)
Com o push por git normal destravado (27/10), o workflow de pytest do
núcleo puro pode finalmente entrar em `.github/workflows/` (o app do
conector não tinha essa permissão). Aceitação: push dispara o CI e o
badge/status aparece no repo; suíte estrutural verde.

## B8 — [DECISÃO DO FUNDADOR] LICENSE do repo
O repo segue sem licença (auditoria 26/10). É decisão de IP, não
tarefa: definir se o projeto é aberto, restrito ou privado antes de
qualquer divulgação externa.
