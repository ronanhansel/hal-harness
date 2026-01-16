FROM python:3.12-slim

# Allow Docker build-time proxy forwarding when the host requires it.
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG http_proxy
ARG https_proxy
ARG no_proxy
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}
ENV NO_PROXY=${NO_PROXY}
ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}
ENV no_proxy=${no_proxy}

# Install system dependencies including headers needed for R package compilation
# These remove environmental barriers where agents fail to compile R packages from source
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    wget \
    libfreetype6-dev \
    libpng-dev \
    libtiff-dev \
    libjpeg-dev \
    libcairo2-dev \
    libxt-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfontconfig1-dev \
    libudunits2-dev \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    cmake \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate base" >> ~/.bashrc \
    && /opt/conda/bin/conda clean -afy

# Add conda to PATH
ENV PATH=/opt/conda/bin:$PATH

# Auto-accept conda ToS so agents can install packages without being blocked
# This removes the "CondaToSNonInteractiveError" environmental barrier
RUN conda config --set plugins.auto_accept_tos true \
 && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
 && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Install mamba for faster dependency solving (10-20x faster than conda)
# Agents can use mamba to install their own packages quickly
RUN conda install -y -n base -c conda-forge mamba \
    && conda clean -afy

# Install minimal core tools that agents need to be ABLE to run R/LaTeX
# This removes barriers where Rscript/pandoc commands don't exist
# Agents are responsible for installing their own R packages
RUN mamba install -y -n base -c conda-forge \
      r-base=4.2 \
      r-biocmanager \
      pandoc \
      texlive-core \
 && conda clean -afy

# Provide a harmless default for optional API-based tools
ENV SERPAPI_API_KEY=dummy-serpapi-key

# Install common Python data science packages
RUN pip install matplotlib numpy pandas scipy scikit-learn seaborn

# Install weave and pin gql < 4, plus wandb version compatible with weave tracing
RUN pip install weave==0.51.41 "gql<4" wandb==0.17.9

# Set working directory
WORKDIR /workspace

# We don't use a non-root user because we need to write files
# that can be read by the host system after the container exits

# Default entry point
CMD ["bash"]
