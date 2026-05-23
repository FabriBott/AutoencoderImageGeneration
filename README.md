# Estructura del proyecto

```text
tarea03-autoencoders/

├── conf/
├── data/
├── notebooks/
├── src/
├── train.py
├── requirements.txt
└── README.md
```

---

# Carpeta conf/

Contiene todas las configuraciones de Hydra (para separar las configuraciones del codigo, y no modificar el codigo fuente).
de modo que todos estos parametros se almacenan dentro de los files .yaml de la carpeta de config

Aquí se define:

- qué modelo utilizar
- qué función de pérdida utilizar
- hiperparámetros
- configuración de WandB
- configuración del entrenamiento

La ventaja es que no es necesario modificar código para ejecutar distintos experimentos.

Ejemplo:

```bash
python train.py model=vae loss=l1
```

o

```bash
python train.py model=unet loss=ssim
```

---

## conf/model/

Define las configuraciones de los modelos.

### vae.yaml

Configuración específica del VAE.

Ejemplo:

```yaml
latent_dim: 128
```

### unet.yaml

Configuración específica del UNet Autoencoder.

---

## conf/loss/

Define la función de pérdida utilizada.

Disponibles:

- L1
- L2
- SSIM
- SSIM + L1

---

## conf/trainer/

Configuración del entrenamiento.

Ejemplo:

```yaml
max_epochs: 50
learning_rate: 0.001
```

---

## conf/logger/

Configuración de WandB.

Permite registrar métricas y visualizaciones.

---

# Carpeta data/

Contiene el dataset MVTec AD.

Actualmente se utilizan únicamente:

- cable
- capsule
- screw
- transistor

Cada clase contiene:

```text
train/
test/
ground_truth/
```

---

# Carpeta notebooks/

Contendrá el notebook final de entrega.

El notebook incluirá:

- explicación del dataset
- descripción de los modelos
- experimentos realizados
- resultados obtenidos
- análisis de reconstrucciones
- histogramas de error
- visualizaciones t-SNE

---

# Carpeta src/

Contiene todo el código fuente.

---

## src/data/

Manejo del dataset.

Archivos:

### mvtec_dataset.py

Este archivo representa el dataset como tal.

Su trabajo consiste en recorrer las carpetas del dataset, localizar las imágenes y convertirlas en ejemplos individuales que puedan ser utilizados durante el entrenamiento.

Responsabilidades:

- abrir imágenes
- aplicar transformaciones
- generar etiquetas
- devolver muestras individuales

---

### mvtec_datamodule.py

Este archivo administra el dataset completo y prepara los datos para el entrenamiento.

Mientras que `MVTecDataset` sabe cómo cargar una imagen individual, `MVTecDataModule` sabe cómo organizar miles de imágenes para entrenar un modelo.

Responsabilidades:

- crear datasets
- crear DataLoaders
- dividir train y validation
- administrar batches

En otras palabras:

es el encargado de alimentar imágenes al modelo.

---

## src/losses/

Funciones de pérdida.

Archivos:

### losses.py

Implementa:

### L1 Loss

Error absoluto promedio.

Pregunta:

"¿Qué tan diferente es cada pixel?"

---

### L2 Loss

Error cuadrático promedio.

Penaliza más errores grandes.

---

### SSIM Loss

Mide similitud estructural.

Pregunta:

"¿La imagen reconstruida mantiene la misma estructura visual?"

---

### SSIM + L1

Combina estructura visual y diferencia por pixel.

---

## src/models/

Arquitecturas neuronales.

---

### vae.py

Implementa el Variational Autoencoder.

Se divide en:

### VAEEncoder

Convierte una imagen en un vector latente.

Imagen:

```text
[3 x 128 x 128]
```

↓

Vector:

```text
[128]
```

---

### Reparameterization

Genera muestras del espacio latente utilizando:

μ (mu)
σ (sigma)

Esto permite generar nuevas imágenes.

---

### VAEDecoder

Reconstruye la imagen desde el vector latente.

Vector:

```text
[128]
```

↓

Imagen:

```text
[3 x 128 x 128]
```

---

### VAE

Une encoder y decoder.

---

## unet_autoencoder.py

Implementa el Autoencoder basado en U-Net.

Componentes:

### Encoder

Reduce progresivamente la imagen.

Extrae características.

---

### Skip Connections

Copian información desde el encoder al decoder.

Permiten conservar detalles visuales.

---

### Decoder

Reconstruye la imagen.

---

### UNetAutoencoder

Combina encoder y decoder completos.

---

## src/lightning/

Contiene la lógica de entrenamiento.

Cuando el entrenamiento comienza, Lightning ejecuta automáticamente un flujo similar a este:

```text
Obtener batch de imágenes
            ↓
Pasar imágenes al modelo
            ↓
Generar reconstrucción
            ↓
Calcular error (loss)
            ↓
Actualizar pesos
            ↓
Guardar métricas
            ↓
Repetir para el siguiente batch
```

El código de esta carpeta le indica a Lightning cómo realizar cada uno de esos pasos.


---

### autoencoder_module.py

LightningModule principal.

Responsabilidades:

- entrenamiento
- validación
- testing
- optimizador
- cálculo de pérdidas
- logging

Es el puente entre:

Dataset
↓
Modelo
↓
Función de pérdida

---

# train.py

Punto de entrada principal del proyecto.

Responsabilidades:

1. Cargar configuración Hydra.
2. Construir DataModule.
3. Construir modelo.
4. Crear Trainer de Lightning.
5. Registrar métricas en WandB.
6. Ejecutar entrenamiento.
7. Ejecutar evaluación.

---

# Flujo completo del sistema

```text
Dataset
↓
DataModule
↓
Modelo
↓
Reconstrucción
↓
Loss
↓
Backpropagation
↓
Actualización de pesos
↓
WandB
```

---

# Experimentos requeridos

VAE

- L1
- L2
- SSIM
- SSIM + L1

U-Net

- L1
- L2
- SSIM
- SSIM + L1

Total:

8 experimentos.

---

# Estado actual

Completado:

- Dataset
- DataModule
- Losses
- VAE
- U-Net Autoencoder
- LightningModule
- Hydra
- WandB
- Entrenamiento básico

Pendiente:

- Comparación completa de experimentos
- Visualización de reconstrucciones
- t-SNE del espacio latente
- Histogramas de error
- Notebook final