FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    HOME=/tmp

RUN apt-get update && apt-get install -y --no-install-recommends \
        fonts-dejavu-core fonts-liberation \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir \
        numpy==2.2.* pillow==11.* scipy==1.15.* sympy==1.13.* \
        matplotlib==3.10.* pandas==2.2.* seaborn==0.13.* squarify==0.4.* \
        networkx==3.4.*

WORKDIR /work
