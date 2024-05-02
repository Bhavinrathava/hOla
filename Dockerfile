FROM python:3.9.19

WORKDIR /usr/src/app

RUN pip install ollama fastapi uvicorn

COPY . .

EXPOSE 8000

CMD ["python" , "ollamaAPI.py"]
