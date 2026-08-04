FROM python:3.11-slim

RUN useradd -m -u 1000 user

USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# =================================================================
# THE MAGIC: ROUTE ALL CACHES TO THE SAFE, WRITABLE USER FOLDER
# =================================================================
ENV HF_HOME=$HOME/app/cache/huggingface \
    NLTK_DATA=$HOME/app/cache/nltk \
    NUMBA_CACHE_DIR=$HOME/app/cache/numba \
    MPLCONFIGDIR=$HOME/app/cache/matplotlib \
    JAX_COMPILATION_CACHE_DIR=$HOME/app/cache/jax

# 1. Copy and install standard requirements first
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy the rest of the engine files
COPY --chown=user . .

# 3. EXECUTE THE TOML: Install the current directory as a native package
RUN pip install --no-cache-dir .

# 4. Ensure cache and data directories exist
RUN mkdir -p $HOME/app/cache/huggingface \
             $HOME/app/cache/nltk \
             $HOME/app/data

EXPOSE 7860

CMD ["python", "app.py"]