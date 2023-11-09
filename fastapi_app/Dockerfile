# Use an official Python runtime as a parent image
FROM python:3.11.4

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY WebApp/requirements.txt /app/

# Install the required packages specified in requirements.txt
RUN pip install -r requirements.txt

# Change directory back to the Docker context before copying the FastAPI application files
# WORKDIR /app/fastapi_app

# Copy the FastAPI application files into the container at /app/fastapi_app
COPY . .

# Expose port 8000
EXPOSE 8000

# Define the command to run your FastAPI application using Uvicorn
CMD ["uvicorn", "app_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
