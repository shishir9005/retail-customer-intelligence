# Runs the dashboard only. The data files in data/processed are built offline (see README).
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY sql ./sql
COPY app ./app
COPY .streamlit ./.streamlit
COPY data/processed ./data/processed

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

# Pass the key at run time:  docker run -p 8501:8501 -e GROQ_API_KEY=... rci
CMD ["streamlit", "run", "app/Overview.py", "--server.port=8501", "--server.address=0.0.0.0"]
