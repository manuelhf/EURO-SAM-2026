# Ouseburn DAFNI model 3: graph learning (IDW vs GNN-topo vs GNN-hydro)
# Build from the repository root:
#   docker build --platform linux/amd64 -f dafni/gnn/Dockerfile -t ouseburn-gnn:latest .
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib HOME=/tmp

WORKDIR /app
COPY dafni/gnn/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu

COPY dafni/common/dafni_io.py dafni/gnn/ouseburn_gnn.py dafni/gnn/run_gnn.py /app/

# DAFNI injects parameters as environment variables, mounts inputs under
# /data/inputs and keeps whatever is written to /data/outputs.
CMD ["python", "/app/run_gnn.py"]
