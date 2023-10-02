# stegano-project
Taller de esteganografía basado en el siguiente paper:

[END-TO-END TRAINED CNN ENCODER-DECODER NETWORKS FOR IMAGE
STEGANOGRAPHY](https://arxiv.org/pdf/1711.07201.pdf)


## Instrucciones para comenzar (setup)

### 1. Hacer git clone del proyecto
Para tener el código que usaremos durante el taller descargar como zip este repositorio o bien usar el siguiente comando:

```shell
git clone https://github.com/Fidu/stegano-project.git
```

Los ejercicios están disponibles en:

- **Ejercicios:** train_model/AI_un_Mensaje_oculto_para_ti.ipynb
- **Soluciones:** train_model/[SOLUCIONES]_AI_un_Mensaje_oculto_para_ti.ipynb


### 2. Configuración en Google Drive del entorno

1. Crearse una carpeta nueva llamada *pycones_2023* en *Mi Unidad* 
2. Acceder a los datos de Drive ([Enlace a los datos de Drive](https://drive.google.com/drive/folders/1dnFs9tIPCAhjgDlZSfuaBRID_oD396jf?usp=sharing)) -> click derecho en *data* -> *Organizar* -> *Añadir acceso directo* y añadirlo en la carpeta creada *pycones_2023*
3. Subir a la carpeta *pycones_2023* el notebook *AI_un_Mensaje_oculto_para_ti.ipynb* (buscar en el directorio train_model)
4. Abrir notebook subido con la aplicación de *Colaboratory* dentro de *Google Drive*.

**Nota:** En caso de tener el navegador o la cuenta de google configurada en inglés habrá que modificar las rutas del notebook a las que se indiquen durante el taller.

## Ejecutar demo con Docker
Para ejecutar la demo habría que seguir los siguientes pasos:

### 1. Construir imagen docker con nuestro Dockerfile

Una vez situados en el mismo directorio que nuestro Dockerfile ejecutar:

```shell
docker build -t inference_api .
```

### 2. Levantar imagen
Ejecutar la imagen exponiendo el puerto *5555*.

```shell
docker run -i -t -p 5555:5555 inference_api
```

# Authors

- Edgar Pérez Sampedro - [LinkedIn](https://www.linkedin.com/in/edgar-p%C3%A9rez-sampedro-a63b68100) - [github](https://github.com/Fidu)

- Javier Jiménez del Peso - [LinkedIn](https://www.linkedin.com/in/javier-jim%C3%A9nez-del-peso-b4559a147) - [github](https://github.com/javijdp)