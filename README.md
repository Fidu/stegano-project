# stegano-project

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