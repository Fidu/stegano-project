# stegano-project
Taller de esteganografía basado en el siguiente paper:

[END-TO-END TRAINED CNN ENCODER-DECODER NETWORKS FOR IMAGE
STEGANOGRAPHY](https://arxiv.org/pdf/1711.07201.pdf)

## Sobre el taller
Los ejercicios están disponibles en:

- [Ejercicios](train_model/AI%20un%20Mensaje%20oculto%20para%20ti.ipynbAI_un_Mensaje_oculto_para_ti.ipynb)
- [Soluciones](train_model/[SOLUCIONES]_AI_un_Mensaje_oculto_para_ti.ipynb)

### Enlace a los datos
[Acceso a los datos en Drive](https://drive.google.com/drive/folders/1Hgg9Mas5tvLcebIUhLBrtn8YON3bQSfF?usp=sharing)

## Ejecutar demo con Docker
Para ejecutar la demo habría que seguir los siguientes pasos:

### 1. Construir imagen docker con nuestro Dockerfile

Una vez situados en el mismo directorio que nuestro Dockerfile ejecutar:

```shell
$ docker build -t inference_api .
```

### 2. Levantar imagen
Ejecutar la imagen exponiendo el puerto *5555*.

```shell
$  docker run -i -t -p 5555:5555 inference_api
```

# Authors

- Edgar Pérez Sampedro - [LinkedIn](https://www.linkedin.com/in/edgar-p%C3%A9rez-sampedro-a63b68100) - [github](https://github.com/Fidu)

- Javier Jiménez del Peso - [LinkedIn](https://www.linkedin.com/in/javier-jim%C3%A9nez-del-peso-b4559a147) - [github](https://github.com/javijdp)