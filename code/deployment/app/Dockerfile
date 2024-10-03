# code/deployment/app/Dockerfile
FROM python:3.8-slim

COPY ./app.py /app/app.py

RUN pip install streamlit requests Pillow

CMD ["streamlit", "run", "/app/app.py"]
