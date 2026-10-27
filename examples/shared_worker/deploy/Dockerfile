# The example's image for compose.yaml, built from a checkout of this
# repository after `mise run build` has put the runtime's wheel in dist/:
#
#   docker build -t shared-worker -f examples/shared_worker/deploy/Dockerfile .
#
# A model built with the `reactor` CLI gets its image from `reactor build`.
FROM python:3.12-slim
COPY dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl
WORKDIR /model
COPY examples/shared_worker /model
CMD ["python", "-m", "reactor_runtime.serve"]
