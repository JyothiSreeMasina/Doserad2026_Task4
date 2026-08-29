# Build from the project root so src/ and configs/ are in the build context:
#   docker build --platform=linux/amd64 -f docker_task3_proton_ct/Dockerfile -t doserad2026_task3_proton_ct .
FROM --platform=linux/amd64 pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime AS task3_proton_ct

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/opt/app

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user docker_task3_proton_ct/requirements.txt /opt/app/
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

COPY --chown=user:user docker_task3_proton_ct/app.py docker_task3_proton_ct/process.py /opt/app/
COPY --chown=user:user src/ /opt/app/src/
COPY --chown=user:user configs/task3_proton_ct.yaml /opt/app/configs/task3_proton_ct.yaml

LABEL org.grand-challenge.api-method="invoke"

ENTRYPOINT ["python", "app.py"]
