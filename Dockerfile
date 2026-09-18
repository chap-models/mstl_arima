# chap_mstl_arima Dockerfile
FROM ghcr.io/dhis2-chap/chapkit-py:latest

# Build steps run as root; the service itself runs as the unprivileged `chapkit`
# user (the switch happens right before CMD). Same posture as chap-core's own
# images. Newer chapkit base images ship the user already (uid/gid 1000); older
# ones do not, so create it only when it is missing.
USER root
RUN id -u chapkit >/dev/null 2>&1 \
    || (groupadd --gid 1000 chapkit && useradd --uid 1000 --gid 1000 --no-create-home --shell /usr/sbin/nologin chapkit)

WORKDIR /work
# Copy lockfile + manifest first so the dep-install layer caches independently of code changes.
COPY pyproject.toml uv.lock ./

# Sync user deps into the venv at /app/.venv. --frozen pins to uv.lock for reproducible
# builds, --no-dev skips the dev-only dependency group, --no-install-project because this
# project is served as a script tree, not installed as a package: the ShellModelRunner
# invokes `python -m chap_mstl_arima` with the copied workspace as cwd.
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_PROJECT_ENVIRONMENT=/app/.venv uv sync --frozen --no-dev --no-install-project

# Git commit the image was built from, exposed as git_revision on /api/v1/info. Same
# convention as chap-core: pass --build-arg GIT_REVISION=$(git rev-parse HEAD) (the
# publish workflow and compose.yml already do). Declared after the uv sync layer so a
# new commit does not invalidate the dependency cache.
ARG GIT_REVISION=""
ENV GIT_REVISION=${GIT_REVISION}

COPY main.py ./
COPY chap_mstl_arima/ ./chap_mstl_arima/

# The service writes in exactly two places: /work/data (SQLite database, a named volume
# in compose.yml) and /tmp (ML workspaces, a tmpfs in compose.yml). Everything else stays
# read-only, which is what compose.yml's `read_only: true` enforces. NUMBA_CACHE_DIR is
# there defensively: numba-backed forecasting stacks write a JIT cache next to the
# installed package, which fails on a read-only root filesystem. statsforecast 2.1.1 does
# not pull numba in, but earlier releases did and a transitive dependency may again.
RUN mkdir -p /work/data && chown -R chapkit:chapkit /work/data
ENV HOME=/tmp \
    MPLCONFIGDIR=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    NUMBA_CACHE_DIR=/tmp/numba_cache
USER chapkit

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

# Run uvicorn directly so server flags (--workers, --log-level, --timeout-keep-alive,
# etc.) are visible here in the Dockerfile. main.py's `if __name__ == "__main__":`
# block stays as the local-dev shortcut for `python main.py`.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
