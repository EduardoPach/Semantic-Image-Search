# Using Official Docker Python Image
FROM python:3.10-slim-bullseye

# Setting working directory
WORKDIR /app

# Copying necessary files and directories
COPY ./frontend .

# Installing dependencies
RUN pip install -r ./requirements.txt

# Exposing port 8501
EXPOSE 8501

# Running streamlit
ENTRYPOINT ["streamlit", "run", "./app.py", "--server.port=8501", "--server.address=0.0.0.0"]