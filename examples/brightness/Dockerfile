# syntax=docker/dockerfile:1
# Reactor model — Dockerfile baseline.
#
# `reactor build` / `reactor run` build and run this image for you; you
# normally don't invoke docker directly. The equivalent docker commands, for
# transparency and one-off debugging:
#
#   docker build -t brightness:dev .
#   docker run --rm -p 8080:8080 brightness:dev

# reactor-runtime ships its own WebRTC media stack as a wheel, so a plain
# Python image is all this build needs — no system media libraries, no build
# toolchain. `reactor build` targets linux/amd64 unless told otherwise.
FROM python:3.12-slim

WORKDIR /app

# requirements.txt pins reactor-runtime and lists the model's own dependencies,
# installed in one pip pass so a conflict fails the build instead of surfacing
# at model load. Installed before the source copy so source-only edits don't
# invalidate the (heavy) pip layer.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app/

# The runtime reads the model out of /app/reactor.yaml and takes no arguments:
# HOST and PORT name the address it serves on. It handles SIGTERM itself and
# spawns no children, so it runs as PID 1 without an init shim.
ENV PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8080
EXPOSE 8080
ENTRYPOINT ["python", "-m", "reactor_runtime.serve"]
