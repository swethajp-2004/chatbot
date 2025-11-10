# Dockerfile — builds Python app + Microsoft ODBC Driver 18 for SQL Server
FROM python:3.13-slim

# Set noninteractive mode to prevent prompts
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

# Add Microsoft repo for ODBC Driver 18 (modern method)
RUN mkdir -p /etc/apt/keyrings \
 && curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > /etc/apt/keyrings/microsoft.gpg \
 && curl -fsSL https://packages.microsoft.com/config/debian/12/prod.list \
      | sed 's#deb https://#deb [signed-by=/etc/apt/keyrings/microsoft.gpg] https://#' \
      > /etc/apt/sources.list.d/mssql-release.list \
 && apt-get update \
 && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 \
 && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy project files into container
COPY . /app

# Upgrade pip and install dependencies
RUN python -m pip install --upgrade pip setuptools wheel

# Install Python dependencies and gunicorn
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi && \
    pip install gunicorn

# Set matplotlib cache directory to prevent runtime issues
ENV MPLCONFIGDIR=/tmp/.matplotlib

# Expose Flask port
ENV PORT=3000
EXPOSE 3000

# Run the Flask app via gunicorn
CMD ["gunicorn", "server:app", "--bind", "0.0.0.0:3000", "--workers", "2"]
