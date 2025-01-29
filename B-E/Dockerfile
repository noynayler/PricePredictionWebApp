
FROM python:3.9-slim


WORKDIR /app

# Copy the requirements file and install dependencies
COPY ../requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


COPY . .

# Set up the virtual environment
RUN python3 -m venv env

# Activate the virtual environment by default and expose the app
ENV VIRTUAL_ENV=/app/env
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Expose the backend port
EXPOSE 5000

# Command to run the app
CMD ["python", "app.py"]
