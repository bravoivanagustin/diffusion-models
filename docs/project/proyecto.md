# TP Final - Cálculo Estocástico - 1er cuatri 2026

Este documento sirve como overview para el proyecto desarrollado para la materia Cálculo Estocástico.

## Objetivo

La idea del trabajo es realizar un trabajo de investigación acerca de una de las áreas propuestas por Patu. Mi idea es llevarlo más para el lado implementación.

## Tema

Dentro de lo que respecta al tema general de lo que me gustaría llevar a cabo están los modelos de difusión, no entiendo bien como, pero muchos de estos aprovechan dinámicas estocásticas para generar imagenes a partir de ruido. 

Por ejemplo, tengo ganas de implementar un sampleador de imagenes de gatos, viendo como distintas utilizaciones o implementaciones de los modulos relacionados con Cálculo Estocástico pueden llevar a distintos resultados. 

Para esto mismo me conviene tener en cuenta modulos dentro de la arquitectura que ya esten implementados y no tengan que ver tanto con lo que quiero presentar $\rightarrow$ Para esto es fundamental primero entender bien la arquitectura.

> **Nota de estado (2026-07-16).** La red (la variable de control del estudio) se construye **a mano**, no se reusa una de librería: el MLP `ScoreMLP` para la Fase 1 (puntos 2D) y la U-Net convolucional `ScoreUNet` para la Fase 2 (imágenes), ambas en `diffusion.models` (decisión del 05/07/2026). Lo estocástico —el forward SDE, el muestreo de pares de entrenamiento y el sampler reverso— es lo que se varía; la red queda fija. El flujo de datos completo entre módulos está en `dataflow.md`.

## Referencias

En el archivo **referencias.md** se pueden encontrar referencias en las cuales voy a basar mi trabajo. La idea es más a las implementaciones clásicas y presentar las bases de lo que es el área hoy en día. 

## Ejes

En el archivo **ejes.md** se puede ver una descripción de los dos caminos a llevar a cabo para el TP, es importante definir bien cuales seria los datasets de entrenamiento. 