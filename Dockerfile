# Using the official Anaconda base image
FROM continuumio/anaconda3:latest

# Setting the working directory
WORKDIR /app

# Copy the environment file
COPY idoven_env.yml .

# Creating and activate a new environment
RUN conda env create -f idoven_env.yml && \
    echo "source activate $(head -1 idoven_env.yml | cut -d' ' -f2)" >> ~/.bashrc && \
    conda clean -afy

# Makeing a RUN command to use the new environment
SHELL ["conda", "run", "-n", "idoven", "/bin/bash", "-c"]

# Installing Jupyter in the conda environment
RUN conda install -c conda-forge jupyterlab

# Expose Jupyter notebook port
EXPOSE 8888

# Coppying the notebook, metadat files, and Data folder into the container
COPY SignalProcessing.ipynb .
COPY ptbxl_database.csv .
COPY data /app/data


# Starting Jupyter notebook
CMD ["jupyter", "lab", "--ip='0.0.0.0'", "--port=8888", "--no-browser", "--allow-root"]

