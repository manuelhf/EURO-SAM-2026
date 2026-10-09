# Ouseburn DAFNI model 1: synthetic sewer network (SWMManywhere)
# Build from the repository root:
#   docker build --platform linux/amd64 -f dafni/network/Dockerfile -t ouseburn-network:latest .
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib HOME=/tmp

# OpenMP runtime for the EPA SWMM engine (swmm-toolkit)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY dafni/network/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY dafni/common/dafni_io.py dafni/network/ouseburn_network.py dafni/network/run_network.py eurosam-ouseburn-gnn/ouseburn_config.yml /app/

# DAFNI injects parameters as environment variables, mounts inputs under
# /data/inputs and keeps whatever is written to /data/outputs.
CMD ["python", "/app/run_network.py"]
