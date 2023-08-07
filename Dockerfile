# Use the official Python base image
FROM python:3.10.6-slim

# Set the working directory inside the container
WORKDIR /app

# Copy your Python API code and requirements file into the container
COPY __init__.py requirements.txt new-model-1.h5 inception_v3_weights_tf_dim_ordering_tf_kernels.h5 ixtoword.pkl wordtoix.pkl  /app/

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt flask

# Expose the port your API will listen on
EXPOSE 5000

# Set the command to run your Python API
CMD ["python", "__init__.py"]
