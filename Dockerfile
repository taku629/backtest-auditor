FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser
ENV PORT=8501
EXPOSE 8501
CMD ["sh", "-c", "streamlit run app.py --server.port ${PORT} --server.address 0.0.0.0"]
