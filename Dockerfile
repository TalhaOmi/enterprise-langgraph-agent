FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir ".[postgres]" \
 && useradd --create-home --uid 10001 app && mkdir /data && chown app /data
USER app
ENV AGENT_AUDIT_LOG_PATH=/data/audit.log.jsonl AGENT_KILL_SWITCH_FILE=/data/.kill_switch
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/healthz')"
CMD ["uvicorn", "ent_agent.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
