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
# ft2build.h (freetype), cmake, libudunits2, etc. are needed when agents try to install R packages from source
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
ENV PATH /opt/conda/bin:$PATH

# Auto-accept conda ToS so agents can install packages without being blocked
# This removes the "CondaToSNonInteractiveError" environmental barrier
RUN conda config --set plugins.auto_accept_tos true \
 && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
 && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Install mamba for faster dependency solving (10-20x faster than conda)
RUN conda install -y -n base -c conda-forge mamba \
    && conda clean -afy

# Preinstall core reproducibility toolchain used by many CoreBench tasks.
# This reduces "environmental barrier" failures where agents are blocked by missing Rscript/pandoc/LaTeX.
# Comprehensive R package installation to overcome environmental barriers.
# Package list derived from analysis of fix packages in fixes/corebench_hard/*/env_override.json
# Using mamba for faster dependency resolution
RUN mamba install -y -n base -c bioconda -c conda-forge \
      # Core R and tools
      r-base=4.2 \
      pandoc \
      jupyter \
      nbconvert \
      texlive-core \
      cmake \
      # Tidyverse and data manipulation
      r-tidyverse \
      r-dplyr \
      r-tidyr \
      r-readr \
      r-stringr \
      r-purrr \
      r-lubridate \
      r-forcats \
      r-tibble \
      r-glue \
      r-stringi \
      r-data.table \
      r-haven \
      r-readxl \
      r-openxlsx \
      r-rio \
      r-reshape \
      r-reshape2 \
      # Visualization
      r-ggplot2 \
      r-ggpubr \
      r-ggrepel \
      r-ggfortify \
      r-ggdist \
      r-cowplot \
      r-patchwork \
      r-scales \
      r-rcolorbrewer \
      r-gridextra \
      r-maps \
      # Statistics and modeling
      r-lme4 \
      r-lmertest \
      r-nlme \
      r-nloptr \
      r-car \
      r-emmeans \
      r-afex \
      r-ez \
      r-effects \
      r-psych \
      r-bayesfactor \
      r-lmtest \
      r-sandwich \
      r-survival \
      r-survminer \
      r-bsts \
      r-estimatr \
      r-lfe \
      r-stargazer \
      # Reporting
      r-knitr \
      r-rmarkdown \
      r-kableextra \
      r-papaja \
      r-xtable \
      # Utilities
      r-matrix \
      r-mass \
      r-devtools \
      r-remotes \
      r-xml2 \
      r-curl \
      r-httr \
      r-jsonlite \
      r-rjson \
      r-yaml \
      r-rvest \
      r-assertthat \
      r-ids \
      r-zoo \
      r-proxy \
      r-hmisc \
      r-rmisc \
      r-rstatix \
      r-rocr \
      r-prroc \
      r-countrycode \
      r-urltools \
      r-tidytext \
      r-reldist \
      r-smoother \
      # Bioconductor packages
      r-biocmanager \
      bioconductor-gsva \
      bioconductor-annotate \
      bioconductor-complexheatmap \
      bioconductor-s4vectors \
      bioconductor-summarizedexperiment \
      bioconductor-survcomp \
      bioconductor-org.hs.eg.db \
      bioconductor-pharmacogx \
      bioconductor-netbiov \
      # Additional specialized packages
      r-snftool \
      r-apcluster \
      r-fingerprint \
      r-bchron \
 && conda clean -afy

# Install packages that may need CRAN fallback
RUN R -e "options(repos = c(CRAN = 'https://cloud.r-project.org')); \
    pkgs <- c('maptools', 'rcdk'); \
    for (pkg in pkgs) { \
      if (!require(pkg, character.only = TRUE, quietly = TRUE)) { \
        tryCatch(install.packages(pkg, dependencies = TRUE), \
          error = function(e) message('Could not install ', pkg, ': ', e\$message)); \
      } \
    }"

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
