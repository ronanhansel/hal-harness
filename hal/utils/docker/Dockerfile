FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    wget \
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
ENV PATH /opt/conda/bin:$PATH

# Preinstall core reproducibility toolchain used by many CoreBench tasks.
# This reduces "environmental barrier" failures where agents are blocked by missing Rscript/pandoc/LaTeX.
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
 && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
 && conda install -y -n base -c conda-forge \
	      r-base=4.* \
	      pandoc \
	      r-knitr \
	      r-rmarkdown \
	      jupyter \
	      nbconvert \
	      texlive-core \
 && conda clean -afy

# Provide a harmless default for optional API-based tools
ENV SERPAPI_API_KEY=dummy-serpapi-key

# Install weave and pin gql < 4, plus wandb version compatible with weave tracing
RUN pip install weave==0.51.41 "gql<4" wandb==0.17.9

# Set working directory
WORKDIR /workspace

# We don't use a non-root user because we need to write files
# that can be read by the host system after the container exits

# Default entry point
CMD ["bash"] 
