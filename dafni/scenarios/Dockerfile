# Ouseburn DAFNI model 2: hydraulic redesign, sensor placement, SWMM scenarios
# Build from the repository root:
#   docker build --platform linux/amd64 -f dafni/scenarios/Dockerfile -t ouseburn-scenarios:latest .
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib HOME=/tmp

# OpenMP runtime for the EPA SWMM engine (swmm-toolkit)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY dafni/scenarios/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY dafni/common/dafni_io.py dafni/scenarios/ouseburn_scenarios.py dafni/scenarios/run_scenarios.py /app/

# DAFNI injects parameters as environment variables, mounts inputs under
# /data/inputs and keeps whatever is written to /data/outputs.
CMD ["python", "/app/run_scenarios.py"]
