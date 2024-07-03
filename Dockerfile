# Use the official Anaconda base image
FROM continuumio/anaconda3:latest

# Set the working directory
WORKDIR /app

# Copy the environment file and the notebook into the container
COPY idoven_env.yml .
COPY SignalProcessing.ipynb .
COPY ptbxl_database.csv .
COPY Data /app/Data

# Create and activate a new environment
RUN conda env create -f idoven_env.yml && \
    echo "source activate $(head -1 idoven_env.yml | cut -d' ' -f2)" >> ~/.bashrc && \
    conda clean -afy

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "idoven", "/bin/bash", "-c"]

# Install Jupyter in the conda environment
RUN conda install -c conda-forge jupyterlab

# Expose Jupyter notebook port
EXPOSE 8888

# Start Jupyter notebook
CMD ["jupyter", "lab", "--ip='0.0.0.0'", "--port=8888", "--no-browser", "--allow-root"]

