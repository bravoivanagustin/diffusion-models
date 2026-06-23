# Estándares Numéricos (SDEs y samplers)

Convenciones de cómputo tensorial compartidas por `sde`, `training` y los futuros samplers. Mantienen
las piezas intercambiables (todas las SDEs cumplen el mismo contrato) y estables cerca de los bordes
temporales.

## Contrato de shapes

- **Estado**: `x`, `x0`, `x_t`, `eps` de shape `(B, data_dim)`.
- **Tiempo**: `t` aceptado como `(B,)` **o** `(B, 1)`; se normaliza a `(B, 1)` con un helper
  (`_expand_t` → `t.reshape(-1, 1)`) para broadcastear sobre las dimensiones.
- **Escalares por muestra** (`std`, `diffusion`): shape `(B, 1)`, se broadcastean sobre `data_dim`.
- **CLD** rompe la forma escalar-gaussiana: `marginal_prob` devuelve `mean (B, 2·spatial)` y un
  **Cholesky** `L (B, 2, 2)` (triangular inferior, diagonal positiva) del kernel conjunto
  posición–momento.

## dtype y device

- Todo en **`float32`** (entradas y salidas). Los tests verifican `dtype == torch.float32`.
- Propagar `device` y `dtype` desde la entrada: `torch.randn(x0.shape, generator=..., device=x0.device,
  dtype=x0.dtype)`. `prior_sampling` toma `device`/`dtype`/`generator` explícitos.

## Reproducibilidad

- Toda aleatoriedad pasa por un `torch.Generator` opcional (`generator=`) o un `seed`. Mismo
  generador → resultado idéntico (`torch.equal`); es la base de la comparabilidad entre celdas.

## Estabilidad en los bordes (`t → 0`, `t → T`)

- **Piso para `std`** antes de dividir: `std.clamp_min(self._std_eps)` con `_std_eps = 1e-5`, para
  evitar la división por cero en `t → 0` (el score `-eps/std` diverge si no).
- Verificar límites con forma cerrada: en VP, `t→0` ⇒ `mean ≈ x0`, `std ≈ 0`; `t→T` ⇒ `mean → 0`,
  `std → 1`. VE sin drift (`mean == x0` exacto). sub-VP con varianza estrictamente por debajo de VP.

## El contrato `ForwardSDE`

Base abstracta con tres métodos a implementar — `sde(x,t) → (drift, diffusion)`,
`marginal_prob(x0,t) → (mean, std)`, `prior_sampling(shape)` — y dos concretos derivados de
`marginal_prob` para la familia escalar-gaussiana:

- `perturb(x0, t) → (x_t, eps)` con `x_t = mean + std·eps`, `eps ~ N(0, I)`.
- `score_target(x0, t, eps) → (score_real, weight)`: para la familia escalar-gaussiana
  `score_real = -eps/std` y `weight = std²` (pesado tipo verosimilitud → pérdida equivalente a
  `‖std·s_θ + eps‖²`).

CLD **sobreescribe** `perturb`/`score_target`/`marginal_prob` (kernel conjunto, target sobre el
momento `∇_v log p_t(v|x)`). Flag `is_augmented` para que los consumidores ramifiquen sin inspeccionar
el tipo.

## Pendiente conocido: pesado HSM de CLD

Hoy `score_target` de CLD devuelve `weight = 1`; sin un pesado adecuado **CLD no converge** (el target
del momento explota con `t → 0`). Decisión abierta: la fórmula del peso y dónde vive (`training` vs
`sde`). Tenerlo presente al tocar la pérdida o agregar samplers que usen CLD.

## Validación numérica

Toda forma cerrada se contrasta de forma independiente: diferencias finitas (ODE de la varianza),
score analítico, o Monte Carlo del forward (Euler–Maruyama, `n` grande). Detalle en
`.kiro/steering/testing.md`.

---
_Convenciones de cómputo, no la implementación de cada SDE (eso vive en `sde/` y `docs/project/sde.md`)._
