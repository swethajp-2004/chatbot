
# Dockerfile — builds Python app + Microsoft ODBC Driver 18 for SQL Server
FROM python:3.11-slim

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies + ODBC driver
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl apt-transport-https gnupg unixodbc-dev build-essential locales && \
    curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
    curl https://packages.microsoft.com/config/debian/12/prod.list > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && ACCEPT_EULA=Y apt-get install -y msodbcsql18 && \
    rm -rf /var/lib/apt/lists/*

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
