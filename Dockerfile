# Use a slim official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# System deps for pyodbc / ODBC driver (Debian-based)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl apt-transport-https unixodbc-dev build-essential \
 && rm -rf /var/lib/apt/lists/*

# Optional: install msodbcsql dependencies (if you plan to use Microsoft ODBC driver in container)
# You may need to add Microsoft's package repo here if you want msodbcsql; leaving commented because different OS images vary.
# RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
#  && curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list \
#  && apt-get update && ACCEPT_EULA=Y apt-get install -y msodbcsql18

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r /app/requirements.txt

# Copy app
COPY . /app

# Expose port used in server.py
EXPOSE 3000

# Use gunicorn to serve flask with multiple workers (better for concurrency)
# server:app — ensures it imports your server.py module and uses app
CMD ["gunicorn", "--bind", "0.0.0.0:3000", "--workers", "4", "--threads", "4", "server:app"]
