FROM python:3.9.16-slim

COPY inference_api /home/inference_api

RUN ["/bin/bash", "-c", "pip install -r /home/inference_api/requirements.txt"]

#Set working directory
WORKDIR /home/inference_api
EXPOSE 5555

CMD ["/bin/bash", "-c", "uvicorn main:app --host 0.0.0.0 --port 5555 --reload"]