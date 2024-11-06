FROM python:3.11-slim

WORKDIR /opt/backend

RUN <<EOF
  apt-get update
  apt-get upgrade -y
  apt-get install -y \
    python3          \
    python3-pip      \
    python3-venv     \

EOF

COPY requirements.txt .

RUN <<EOF
  python3 -m venv .venv
  source .venv/bin/activate
  pip3 install -r requirements.txt
EOF

COPY backend.py .

ENV MLFLOW_HOST="host.docker.internal"
ENV MLFLOW_PORT="8080"

ENV FLASK_RUN_HOST="0.0.0.0"
ENV FLASK_RUN_PORT="8715"

ENTRYPOINT ["flask", "--app", "backend", "run"]
