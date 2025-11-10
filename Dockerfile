# Dockerfile — builds Python app + Microsoft ODBC Driver 18 for SQL Server
FROM python:3.13-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies required for pyodbc & msodbcsql
RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg \
        wget \
        build-essential \
        unixodbc-dev \
        locales \
    && rm -rf /var/lib/apt/lists/*

# Install Microsoft package signing key and repo (write key into /usr/share/keyrings)
RUN set -eux; \
    mkdir -p /usr/share/keyrings; \
    curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg; \
    echo "deb [signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" \
      > /etc/apt/sources.list.d/mssql-release.list; \
    apt-get update; \
    ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18; \
    rm -rf /var/lib/apt/lists/*

# Create app dir
WORKDIR /app

# Copy project files into container
COPY . /app

# Upgrade pip and install dependencies
RUN python -m pip install --upgrade pip setuptools wheel

# Install Python dependencies and gunicorn
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi && \
    pip install gunicorn

# Prevent matplotlib font cache issues at runtime
ENV MPLCONFIGDIR=/tmp/.matplotlib

ENV PORT=3000
EXPOSE 3000

CMD ["gunicorn", "server:app", "--bind", "0.0.0.0:3000", "--workers", "2"]
