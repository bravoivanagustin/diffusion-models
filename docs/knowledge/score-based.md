# Score-Based Diffusion Models

"Score-Based Generative Modeling through Stochastic Differential Equations" es el paper que complementa el area de modelos de difusión, basandose en Ecuaciones Diferenciales Estocasticas para tener dinamicas continuas en vez de discretas. 

## Idea principal

Como proceso es parecido a DDPM, pero donde los pasos son lo suficientemente chicos como para considerarlos continuos. La idea es que se cumple que:

$$dx = f(x,t)dt+g(t)dW$$

Donde $W$ representa un movimiento browniano. $f$ se llama drift, comanda la dirección general, y $g$ se llama diffusion coefficient, agrega ruido a los recorridos de las particulas.

La ecuancion de DDPM se puede transformar a una ecuacion diferencial estocastica:

$$dx = -\frac{1}{2}\beta(t)x(t)dt+\sqrt{\beta(t)}dW$$

## Distribucion

Si tomamos una distribución de puntos entonces el proceso que cumple DDPM pero de manera continua transforma la distribución a una normal estandar. 

La idea es ahora encontrar una transformación inversa al ruido que se agrego para obtener la distribución original. 

## Anderson

Tiene un paper donde se demustra que bajo ciertas condiciones se tiene una ecuacion diferencial estocastic inversa para obtener la distribución original (buscar y ver demostración).

## Reverse SDE

Se tiene como SDE inversa a: 

$$dx = [f(x,t)+g(t)^2\nabla_x log(p_t(x))]dt + g(t)d\bar{W}$$

Donde $\nabla_x log(p_t(x))$ es un campo que apunta en la dirección de crecimiento de la densidad de la probabilidad. O sea, apunta a zonas mas probables. A este campo se le llama **score** y es en lo que se basa este metodo. 

## Aprendizaje de $\nabla_x log(p_t(x))$

La idea de la red neuronal que usemos tiene que mejorar $s_\theta(x,t)$ para se parezca todo lo posible a $\nabla_x log(p_t(x))$. 

Como lo entrenamos? Minimizamos $\mathcal{L}(\theta) = \mathbb{E}_{x_0\sim p(x), t}[\left\|s_\theta(x,t)- \nabla_x log(p_t(x))\right\|^2]$.

Podemos reemplazar $\nabla_x log(p_t(x))$ por $\frac{\epsilon}{\beta_t}$, ya que la primera es desconocida. 

Ahora que tenemos un buena aproximacion del score podemos discretizar la SDE inversa para un $x_T$ cualquiera y tenemos una muestra de una distribucion parecida a $p(x)$.

## Generalidad

DDPM es un caso particular de Score-Based, y hay varias maneras de implementar el Score Matching, por ejemplo Hyvärinen y Sliced. 

### Nota

Estas notas se tomaron del video **https://www.youtube.com/watch?v=lUljxdkolK8**, se puede investigar mas leyendo el paper. 