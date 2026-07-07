# Brief: train-decoupling

## Problem

El `train()` actual construye la red por dentro (`ScoreMLP(...)` hardcodeado) y consume un
`distribution` finito vía `dataloader`. Eso impide entrenar la `ScoreUNet` de Fase 2 (u otra red)
con el mismo loop: la red está clavada al MLP y el loop está atado a un dataset finito de puntos. Es
el acople que la spec `score-unet` dejó anotado como pendiente (ver roadmap, "Shared seams to watch").

## Current State

- `training/trainer.py`: `train(sde, distribution, config)` instancia `ScoreMLP` con hiperparámetros
  tomados de `TrainConfig` (`embed_dim`, `hidden_dim`, `num_blocks`, `activation`), arma
  `loader = distribution.dataloader(n, batch_size, shuffle=True)` y corre un doble `for` sobre
  épocas × batches, desempaquetando `(x0,)`. `history` es por época.
- `TrainConfig` carga hiperparámetros de red (`embed_dim/hidden_dim/num_blocks/activation`) y de datos
  (`epochs`, `n_samples`, `batch_size`) mezclados con los del loop (`lr/grad_clip/t_eps/device/seed/log_every`).
- **Checkpoints (consumidor vivo):** `save_checkpoint` lee los hiperparámetros de red de `TrainConfig`
  y `result.net.data_dim`; `load_checkpoint` **reconstruye una `ScoreMLP`** desde esa metadata.
  `samplers/generate.py:87` **depende** de `load_checkpoint` para la generación checkpoint-driven
  (con tests en `test_samplers.py`). Quitar esos campos de `TrainConfig` rompe ambos.
- **Config-driven (YAML):** `config.py` `build_run`/`RunSpec` ensambla `(sde, distribution, TrainConfig)`
  y funnelea un bloque `model:` a los campos de red de `TrainConfig`; `scripts/train.py` llama
  `train(spec.sde, spec.distribution, spec.config)` y `save_checkpoint`. Ambos rompen con la firma nueva.
- `data_generation`: `PointDistribution.dataloader(n, batch_size, *, shuffle=True)` devuelve un loader
  finito que yield-ea `(x0,)`. No hay adaptador infinito ni pipeline de imágenes.
- `sample_timesteps`, `dsm_loss`, el paso de optimización, el grad-clip y la lógica de generator/seed
  ya son agnósticos y NO cambian.

## Desired Outcome

`train()` recibe la red ya construida y una fuente de datos como **iterador infinito de tensores
crudos**, con un loop **por pasos** (no por épocas). El mismo `train()` entrena MLP (puntos) o U-Net
(imágenes) sin ramificar. La suite queda en verde en cada paso, sin regresiones (checkpoints y CLI
siguen funcionando). Se verifica corriendo el MLP con la firma nueva sobre swiss roll y confirmando
que la pérdida baja como antes (comparar **tendencia**, no valores paso a paso — cambia el orden de
consumo de ruido al desaparecer el borde de época).

Firma objetivo:

```python
def train(
    sde: ForwardSDE,
    model: ScoreModel,     # ya construida; train no sabe si es MLP o U-Net
    data,                  # iterador infinito que yield-ea tensores (B, ...)
    config: TrainConfig,
    *,
    generator: torch.Generator | None = None,
) -> TrainResult:
    net = model.to(device); net.train()   # .to idempotente
    ...
    for step in range(config.num_steps):
        x0 = next(data_iter).to(device)    # tensor crudo, sin (x0,)
        ...
```

## Approach

Refactor del módulo `training` en una sola spec, con las dos decisiones de frontera resueltas en
discovery (07/07/2026) del lado **sin regresión**:

1. **`train()` + loop por pasos**: firma nueva (`model` + `data`), `net = model.to(device)` (idempotente),
   `for step in range(config.num_steps)` con `x0 = next(data_iter).to(device)` (tensor crudo, sin
   unpack), `history` por intervalo de logging.
2. **`TrainConfig` adelgazado**: quita `epochs`, `n_samples`, `batch_size`, `embed_dim`, `hidden_dim`,
   `num_blocks`, `activation`; agrega `num_steps`. Quedan `lr`, `grad_clip`, `t_eps`, `device`, `seed`,
   `log_every`. `TrainResult` igual (solo cambia que `history` es por intervalo).
3. **Adaptador `infinite_bare`** en `data_generation`: envuelve un `DataLoader` finito para hacerlo
   infinito y yield-ear el tensor crudo (`while True: for (x0,) in loader: yield x0`). El swiss roll
   sigue usando `distribution.dataloader(...)`, solo envuelto.
4. **Checkpoints model-agnósticos** (decisión 1): `save_checkpoint` guarda `state_dict` + `sde_name` +
   `history` (sin hiperparámetros de red); `load_checkpoint` devuelve `(state_dict, meta)` y el
   **caller** reconstruye la red y carga el state. Actualizar `samplers/generate.py` para construir su
   red y cargar el state, y sus tests, de modo que la generación checkpoint-driven siga funcionando.
5. **Config-driven actualizado** (decisión 2): `build_run` construye también el modelo (desde el bloque
   `model:`) y el iterador de datos (`dataloader` envuelto en `infinite_bare`); `RunSpec` pasa a llevar
   `model` + `data`; `scripts/train.py` y los tests de config se actualizan. El CLI YAML sigue andando
   de punta a punta.
6. **`num_steps`**: no hay traducción automática desde épocas. Elegir para swiss roll un `num_steps`
   ≈ `epochs × (n_samples / batch_size)` de la corrida vieja, para que el MLP entrene una cantidad
   comparable y la comparación sea justa (detalle de tarea, no de frontera).

## Scope

- **In**: `training/trainer.py` (`train`, `TrainConfig`, `TrainResult` doc, `save_checkpoint`,
  `load_checkpoint`), el adaptador `infinite_bare` en `data_generation`, `config.py`
  (`build_run`/`RunSpec` + esquema YAML), `scripts/train.py`, `samplers/generate.py` (adaptación al
  nuevo contrato de checkpoint), y las suites afectadas (`test_training.py`, `test_samplers.py`, y
  cualquier test de `data_generation` que cubra el adaptador). Doc del módulo `training.md` actualizada.
- **Out**: pipeline de datos de imágenes / `infinite_batches` / dataset de gatos-CIFAR-FashionMNIST
  (sigue "a definir", spec futura); EMA de pesos (el "segundo paso" que mencionó el autor, aparte);
  cualquier cambio a `ScoreMLP`/`ScoreUNet`/`layers`/`sde` más allá de imports; la evaluación /
  visualización de Fase 1.

## Boundary Candidates

- El loop y la firma de `train()` + `TrainConfig` (núcleo).
- El contrato de checkpoint (`save`/`load`) y su consumidor `samplers/generate.py`.
- El front-end config-driven (`config.py`, `scripts/train.py`, esquema YAML).
- El adaptador de datos `infinite_bare` (en `data_generation`).

## Out of Boundary

- Construir el pipeline de imágenes o `infinite_batches` (no hay dataset de imágenes todavía).
- Introducir EMA (paso posterior).
- `make_model`/registry de redes: el caller construye la red explícitamente; no se añade factory salvo
  que el config-driven lo necesite mínimamente (decidir en diseño, preferir YAGNI).

## Upstream / Downstream

- **Upstream**: `diffusion.models` (`ScoreModel` como tipo del parámetro `model`; `ScoreMLP`/`ScoreUNet`
  como redes que el caller construye), `diffusion.sde` (`ForwardSDE`), `dsm_loss`/`sample_timesteps`
  (sin cambios), `data_generation.PointDistribution.dataloader` (base del `infinite_bare`).
- **Downstream**: la spec futura de dataset/entrenamiento de imágenes (usará el mismo `train()` con
  `infinite_batches`), y el "segundo paso" de EMA + checkpoints enriquecidos.

## Existing Spec Touchpoints

- **Extends**: ninguna spec formal (el módulo `training` no tiene spec propia; se creó como item de
  desarrollo). Esta es su primera spec.
- **Adjacent**: `samplers` (completa) — su `generate.py` consume `load_checkpoint`; esta spec cambia
  ese contrato y debe actualizar `generate.py` + sus tests sin romperlos. `score-unet` (completa) —
  provee `ScoreUNet`/`ScoreModel`; no se toca, solo se consume.

## Constraints

- **Sin regresiones, suite en verde en cada paso** (convención del repo): checkpoints y CLI deben
  seguir funcionando; los tests que hoy los cubren se adaptan, no se rompen.
- **`train()` agnóstico a la red** — no debe importar `ScoreMLP`/`ScoreUNet` ni ramificar por tipo;
  tipa `model: ScoreModel` (Protocol estructural de `models.base`).
- **Reproducibilidad**: la lógica de `generator`/`seed` se preserva idéntica; verificación por
  **tendencia** de pérdida (el orden de consumo de ruido cambia al quitar el borde de época).
- **Stack**: Python 3.14 + torch 2.12 CPU; `importorskip("torch")` en tests; docstrings en español.
