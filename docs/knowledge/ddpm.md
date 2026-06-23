# Diffusion Probabilistic Models

"Denoising Diffusion Probabilistic Models" es el paper que presentó los modelos de difusión de una manera mas profesional y rigurosa, es considerado el nacimiento del área.

## Idea principal

Tratar las imagenes como si tuvieran una distribución de probabilidad subyacente. Para descubrir esta distribución se entrena un modelo que, dada una cadena de ruido agregado a una imagen, sabe como reconstruir la imagen original, o sea, sacar ruido. 

La idea es que la probabilidad de una imagen $x_1$ dada la imagen original $x_0$ es una normal centrada en $x_0$ con una determinada varianza por un muestra de una normal. O sea, le agregamos ruido gaussiano. 

El problema es que el agregando varianza siempre no llegamos a una distribución "linda", como una normal estandar. A esto se lo conoce como Variance Exploding Diffusion. 

## Probabilidad

Se toma $q(x_t|x_0) = \sqrt{\bar{\alpha_t}}x_0 + (1-\bar{\alpha_t})\epsilon$. Donde $\bar{\alpha_t}=\prod_{t=1}^{T}{(1-\beta_i)}$ y $\epsilon \sim N(0, I)$.

La idea es encontrar $\theta$ tal que $p_\theta(x_{t-1}|x_t)$ para eliminar el ruido efectivamente. Objetivo: Minizar la negative log-likelihood.

Usando que es un proceso de Markov tenemos que: 
$$q(x_{1:T}|x_0) = \prod_{t=1}^{T}q(x_t|x_{t-1})$$
$$p_\theta(x_{0:T}) = p_\theta(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1}|x_{t})$$

## Función objetivo

Usando igualdades y la desigualdad de Jensen tenemos que
$$\mathbb{E}\left[-log(p_\theta(x_0))\right]\leq\mathbb{E}_{q(x_{1:T}|x_0)}\left[-log\left(\frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right)\right]$$
$$=\mathbb{E}_q\left[D_{KL}(q(x_T|x_0)\;||\;p(x_T))+\sum_{t>0}D_{KL}(q(x_{t-1}|x_t,x_0)\;||\;p_\theta(x_{t-1}|x_t))-log(p_\theta(x_0|x_1))\right]$$

El primer termino no depende de $\theta$ y el ultimo es insignificante, asi que se eliminan de la funcion objetivo.

Además $q(x_{t-1}|x_t,x_0)$ es conocida, por lo cual podemos entrenar un modelo para minimizar el termino central en función de $\theta$. Se fija $p_\theta(x_{t-1}|x_t)$ como una normal, hay que buscar minimizar la diferenciar entre dos normales, de esta primera se fija la varianza, por lo cual hay que minizar la diferencia de las medias.

Se puede hacer que esta ultima solo dependa de $x_t$ y $\epsilon$, por lo cual al final lo que tiene que aprender el modelo es, dado un $x_t$ y $t$, como predigo el ruido que se le agregó en ese momento?

## Simplificación final

Nos quedo una suma de mucho tiempo para cada sample, eso es costoso de computar, la idea es para cada sample tomar un tiempo aleatorio y solo computar esa perdida. Para una cantidad grande de samples esta perdida es suficiente para que el modelo aprenda de manera rigurosa el ruido. 

### Nota

Estas notas se tomaron del video **https://www.youtube.com/watch?v=EhndHhIvWWw&t=410s**, se puede investigar mas leyendo el paper. Lo lei antes del video y no se entendió mucho. 