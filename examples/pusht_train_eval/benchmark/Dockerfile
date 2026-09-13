# The benchmark side for --docker: vla-eval, the env, and the adapter. No GPU needed.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 libxcb1 \
    && rm -rf /var/lib/apt/lists/*  # opencv (via gym-pusht) needs the last three
RUN pip install --no-cache-dir "gym-pusht==0.1.6" "pymunk<7" \
    "vla-eval @ git+https://github.com/allenai/vla-evaluation-harness@cf74191a4b3b53c135ffa525f9654c246c090be3"
WORKDIR /workspace
COPY __init__.py pusht.py benchmark/
ENV PYTHONPATH=/workspace SDL_VIDEODRIVER=dummy
ENTRYPOINT ["vla-eval"]
