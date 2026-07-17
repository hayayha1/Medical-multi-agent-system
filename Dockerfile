FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY app ./app
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN pip install --no-cache-dir .
COPY migrations ./migrations
COPY scripts ./scripts
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
