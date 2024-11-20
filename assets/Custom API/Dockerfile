# Use a base image with Python installed
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the Flask application files to the container
COPY . /app


# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8080
EXPOSE 3005

# Run the Flask application
CMD ["flask", "--app", "translate_endpoint", "run", "-p", "3005"]