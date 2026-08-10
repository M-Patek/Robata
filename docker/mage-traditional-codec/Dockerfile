FROM mwader/static-ffmpeg:7.1.1@sha256:11a44711684c0b9f754c047dcd64235b8b52deab251bd0e0a86f22faa160749c AS ffmpeg

FROM python:3.12-slim@sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

COPY --from=ffmpeg /ffmpeg /usr/local/bin/ffmpeg
COPY --from=ffmpeg /ffprobe /usr/local/bin/ffprobe

COPY requirements.lock /opt/robata/requirements.lock
RUN python -m pip install --require-hashes --only-binary=:all: \
    --requirement /opt/robata/requirements.lock

COPY toolchain.json /opt/robata/toolchain.json
COPY run_cv_preinfer.py /opt/robata/run_cv_preinfer.py

ENTRYPOINT ["python", "/opt/robata/run_cv_preinfer.py"]