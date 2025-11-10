# Dockerfile — builds Python app + Microsoft ODBC Driver 18 for SQL Server
FROM python:3.13-slim

# install system deps needed for pyodbc & msodbcsql
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
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
 && curl https://packages.microsoft.com/config/debian/12/prod.list > /etc/apt/sources.list.d/mssql-release.list \
 && apt-get update \
 && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 \
 && rm -rf /var/lib/apt/lists/*

# create app dir
WORKDIR /app

# copy project files
COPY . /app

# ensure pip, install python deps
RUN python -m pip install --upgrade pip setuptools wheel

# Install requirements if file exists, then ensure gunicorn present
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi && \
    pip install gunicorn

# optional: ensure font cache build does not fail at runtime (matplotlib)
ENV MPLCONFIGDIR=/tmp/.matplotlib

# expose port
ENV PORT=3000
EXPOSE 3000

# run the server with gunicorn (same command you used on Render)
CMD ["gunicorn", "server:app", "--bind", "0.0.0.0:3000", "--workers", "2"]
