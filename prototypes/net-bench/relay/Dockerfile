# One direction of the benchmark's emulated link: a WebSocket relay that netem shapes on this
# container's own interface. Two run side by side, one per direction. iproute2 provides tc (netem).
FROM python:3.12-slim@sha256:3d5ed973e45820f5ba5e46bd065bd88b3a504ff0724d85980dcd05eab361fcf4
RUN apt-get update \
 && apt-get install -y --no-install-recommends iproute2 \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir websockets==17.3
COPY relay.py /relay.py
EXPOSE 8765
CMD ["python", "-u", "/relay.py"]
