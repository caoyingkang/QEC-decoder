FROM debian:bookworm
RUN apt-get update
RUN apt-get install -y git git-lfs curl
RUN apt-get install -y python3 python3-pip python3-venv

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

RUN python3 -m venv /opt/venv
SHELL ["/bin/bash", "-lc"]
RUN source /opt/venv/bin/activate && pip install maturin

