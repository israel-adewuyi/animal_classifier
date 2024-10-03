# code/deployment/api/Dockerfile
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

COPY ./main.py /app/main.py
# COPY ./models/animal_classifier.h5 /app/models/animal_classifier.h5
COPY ./models/animal_classifier.pth /app/models/animal_classifier.pth

RUN pip install torch Pillow

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
