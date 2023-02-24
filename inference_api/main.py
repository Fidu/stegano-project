import io
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image
from skimage.io import imread
from stegano_model.model_utils import load_model, inference_encoder, inference_decoder


app = FastAPI()

PRETRAINED_MODEL = f"data/models/pretrained_model.pt"
PRETRAINED_MODEL_2 = f"data/models/pretrained_model_2.pt"
SIZE_IMAGE = 64
model_pretrained = load_model(PRETRAINED_MODEL)
model_pretrained_2 = load_model(PRETRAINED_MODEL_2)


@app.get("/")
def read_root():
    html_content = """
    <html>
        <head>
            <title>DEMO PRODUCTIVIZACIÓN</title>
        </head>
        <body>
             <h1><a href="http://0.0.0.0:5555/docs"> Acceder a la DEMO </a></h1>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/file/model_1/encoder")
def encoded_predict(file_source: UploadFile, file_payload: UploadFile):
    encoded_result_path = inference_encoder(
        model=model_pretrained,
        path_source=file_source.file,
        path_payload=file_payload.file,
        device="cpu",
    )

    return FileResponse(encoded_result_path, media_type="application/octet-stream", filename=f"1_encoded_{encoded_result_path}")

@app.post("/file/model_1/decoder")
def decode_predict(file_encoded: UploadFile):
    decoded_result_path = inference_decoder(
        model=model_pretrained,
        path_image_to_decode=file_encoded.file,
        device="cpu",
    )

    return FileResponse(decoded_result_path, media_type="application/octet-stream", filename=f"1_decoded_{decoded_result_path}")

@app.post("/file/model_2/encoder")
def encoded_predict(file_source: UploadFile, file_payload: UploadFile):
    encoded_result_path = inference_encoder(
        model=model_pretrained_2,
        path_source=file_source.file,
        path_payload=file_payload.file,
        device="cpu",
    )

    return FileResponse(encoded_result_path, media_type="application/octet-stream", filename=f"2_encoded_{encoded_result_path}")

@app.post("/file/model_2/decoder")
def decode_predict(file_encoded: UploadFile):
    decoded_result_path = inference_decoder(
        model=model_pretrained_2,
        path_image_to_decode=file_encoded.file,
        device="cpu",
    )

    return FileResponse(decoded_result_path, media_type="application/octet-stream", filename=f"2_decoded_{decoded_result_path}")
