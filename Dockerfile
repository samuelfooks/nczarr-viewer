FROM python:3.12-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libproj-dev \
    proj-data \
    proj-bin \
    libgeos-dev \
    libgdal-dev \
    gdal-bin \
    libnetcdf-dev \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Enable bytecode compilation and set link mode to copy
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies without the project code first
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Copy the rest of the application code
COPY . .

# Install the application into the virtual environment
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Update PATH to use the virtual environment's binaries
ENV PATH="/app/.venv/bin:$PATH"

# Expose the application's port
EXPOSE 8050

# Run the application
ENTRYPOINT ["python", "-u", "src/zarr_data_viewer.py"]
