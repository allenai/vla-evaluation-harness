# Environment image for --docker: vla-eval, gym-pusht, and the adapter. No GPU needed.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 libxcb1 \
    && rm -rf /var/lib/apt/lists/*  # opencv (via gym-pusht) needs the last three
RUN pip install --no-cache-dir "gym-pusht==0.1.6" "pymunk<7" \
    "vla-eval>=0.6.0"
WORKDIR /workspace
COPY __init__.py pusht.py benchmark/
ENV PYTHONPATH=/workspace SDL_VIDEODRIVER=dummy
ENTRYPOINT ["vla-eval"]
