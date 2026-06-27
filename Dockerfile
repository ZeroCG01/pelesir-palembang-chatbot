FROM python:3.10-slim

# Set working directory
WORKDIR /code

# Copy requirements
COPY ./requirements.txt /code/requirements.txt

# Install dependencies
# Kita tetap gunakan CPU version agar ukuran image (Docker) tidak membengkak
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Memberikan akses pada user non-root (syarat wajib di Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy seluruh source code dengan ownership ke user
COPY --chown=user . $HOME/app

# Hugging Face Spaces mengekspos port 7860 secara default
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
