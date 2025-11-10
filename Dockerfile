
# Dockerfile — builds Python app + Microsoft ODBC Driver 18 for SQL Server
FROM python:3.11-slim
# Install system deps needed for pyodbc & msodbcsql
ENV DEBIAN_FRONTEND=noninteractive
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

# Add Microsoft package signing key and repo for ODBC Driver 18 (Debian/Ubuntu)
RUN mkdir -p /etc/apt/keyrings \
 && curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > /etc/apt/keyrings/microsoft.gpg \
 && curl -fsSL https://packages.microsoft.com/config/debian/12/prod.list \
      | sed 's#deb https://#deb [signed-by=/etc/apt/keyrings/microsoft.gpg] https://#' \
      > /etc/apt/sources.list.d/mssql-release.list \
 && apt-get update \
 && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt
RUN pip install gunicorn pyodbc sqlalchemy pandas matplotlib openai

# Expose port
ENV PORT=3000
EXPOSE 3000

# Run app
CMD ["gunicorn", "server:app", "--bind", "0.0.0.0:3000", "--workers", "2"]



