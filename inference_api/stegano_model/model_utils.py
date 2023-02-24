import torch
import numpy as np
from PIL import Image
from stegano_model.steg_net import StegNet
from skimage.io import imsave, imread

def load_model(path_model):
    """
    Metodo para cargar los pesos de un modelo preentrenado
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = StegNet()
    model.load_state_dict(
    torch.load(path_model, map_location={'cuda:0': 'cpu'}))  # Choose whatever GPU device number you want
    model.to(device)
    return model

def open_image_and_resize(image, width, height):
    # TODO: redimensionar imagen a tamaño 64x64
    im = Image.open(image)
    newsize = (width, height)
    image_resized = im.resize(newsize)
    return image_resized

def image_to_tensor(path_image, convert="RGB", device="cpu"):
    image_array = np.array(open_image_and_resize(path_image, 64, 64).convert(convert)).astype(np.uint8)
    return torch.from_numpy(image_array).float().to(device)


def images_to_tensor(path_source, path_payload, device='cpu'): #todo nosotoros: cambiar método para llamar un open and resize 
    image_source_tensor = image_to_tensor(path_source, convert="RGB", device=device)
    image_payload_tensor = image_to_tensor(path_payload, convert="L", device=device)

    return image_source_tensor, image_payload_tensor


def inference_encoder(model, path_source, path_payload, size_image=64, device='cpu'):
    im_source, im_payload = images_to_tensor(path_source, path_payload, device=device)
    with torch.no_grad():
        model.eval()
        encoded_image = model.predict_encoder(im_source, im_payload) 

    encoded_image = encoded_image.cpu()
    numpy_tensor_encoded = encoded_image.view((-1, size_image, size_image, 3)).numpy()[0]

    path_im_encoded = 'results/image_encoded_result.tiff'
    imsave(path_im_encoded, numpy_tensor_encoded/255)

    return path_im_encoded


def inference_decoder(model, path_image_to_decode, size_image=64, device='cpu'):
    # image_to_decode = Image.frombytes(size=(64,64), data=path_image_to_decode.read(), mode="RGB")
    with open('.file_temporary', 'wb') as f: 
        f.write(path_image_to_decode.read())
    image_to_decode = imread('.file_temporary')
    # print(image_to_decode)
    image_to_decode = image_to_decode.astype(np.float32)*255
    # print(image_to_decode)
    image_to_decode_tensor = torch.from_numpy(image_to_decode).float().to(device)

    with torch.no_grad():
        model.eval()
        decoded_payload = model.predict_decoder(image_to_decode_tensor)

    decoded_payload = decoded_payload.cpu()
    decoded_payload_result = decoded_payload.view((-1, size_image, size_image)).numpy()[0].astype('uint8')
    
    path_im_decoded = 'results/image_decoded_result.png'
    imsave(path_im_decoded, decoded_payload_result)

    return path_im_decoded
