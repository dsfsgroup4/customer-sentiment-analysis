# Use miniconda base image
FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Copy environment config and app files
COPY . /app

# Create conda environment and install dependencies
RUN conda update -n base -c defaults conda && \
    conda create -n app_env python=3.10 -y && \
    echo "conda activate app_env" >> ~/.bashrc

# Activate conda env, install pip packages
RUN /bin/bash -c "source ~/.bashrc && conda activate app_env && \
    pip install --upgrade pip && \
    pip install streamlit pandas plotly matplotlib seaborn langchain \
    langchain-mistralai python-dotenv"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Expose the port for Hugging Face
EXPOSE 7860

# Run your Streamlit app
CMD ["/bin/bash", "-c", "source ~/.bashrc && conda activate app_env && streamlit run app.py"]
