# Use an official Python runtime as a parent image
FROM python:3.11.4

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY WebApp/requirements.txt /app/

# Install the required packages specified in requirements.txt
RUN pip install -r requirements.txt

# Copy the Streamlit application files into the container at /app
COPY . .

# Expose port 8501 for Streamlit app
EXPOSE 8501

# Define the command to run your Streamlit application
CMD ["streamlit", "run", "app_streamlit.py"]
