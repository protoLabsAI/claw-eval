FROM ghcr.io/agent-infra/sandbox:latest

WORKDIR /workspace

# Install Python dependencies first (better layer caching)
COPY pyproject.toml .
RUN pip install -e ".[mock,web]" --no-cache-dir 2>/dev/null || \
    pip install -e ".[mock]" --no-cache-dir

# Copy source code and mock services
COPY src/ src/
COPY mock_services/ mock_services/

# Re-install in editable mode with full source
RUN pip install -e ".[mock]" --no-cache-dir

# Create trace output directory
RUN mkdir -p /workspace/traces

ENTRYPOINT ["claw-eval"]
CMD ["--help"]
