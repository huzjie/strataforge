# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml setup.py README.md ./
COPY strataforge ./strataforge
COPY configs ./configs

RUN pip install --no-cache-dir .

EXPOSE 8080

ENV STRATA_BACKEND=mock
CMD ["python", "-m", "strataforge", "serve", "--host", "0.0.0.0", "--port", "8080"]
