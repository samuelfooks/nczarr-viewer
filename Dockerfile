# Stage 1: Build dependencies and install the application
FROM python:3.12-slim-bookworm AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Enable bytecode compilation and set link mode to copy
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Copy dependency files first to leverage Docker layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies without the project code
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Copy the rest of the application code
COPY . .

# Install the application into the virtual environment
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Stage 2: Create a minimal runtime image
FROM python:3.12-slim-bookworm

# Install runtime system dependencies for cartopy and other geospatial libraries
RUN apt-get update && apt-get install -y \
    libproj-dev \
    proj-data \
    proj-bin \
    libgeos-dev \
    libgdal-dev \
    gdal-bin \
    libnetcdf-dev \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the virtual environment and application code from the builder stage
COPY --from=builder /app /app

# Update PATH to use the virtual environment's binaries
ENV PATH="/app/.venv/bin:$PATH"

# Expose the application's port
EXPOSE 8050

# Define the default command to run the application
ENTRYPOINT ["python", "-u", "src/zarr_data_viewer.py"]
