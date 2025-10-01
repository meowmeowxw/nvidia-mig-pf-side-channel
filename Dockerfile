FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

RUN apt update && apt install -y python3 python3-pip python3.12-venv

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace
