# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app
# Keep CUDA dependencies independent of runtime and model source layers.
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
COPY runtime-wheel/ /tmp/runtime-wheel/
RUN pip install --no-cache-dir -r /tmp/runtime-wheel/requirements.txt
COPY model.py reactor.yaml /app/
EXPOSE 8080
ENTRYPOINT ["python", "-m", "reactor_runtime.serve"]
CMD ["run"]
