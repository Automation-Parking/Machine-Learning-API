FROM python:3.11.4
ENV PYTHONBUFFERED True
ENV APP_HOME /app
WORKDIR $APP_HOME
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libopencv-dev \
    opencv-python \
    && rm -rf /var/lib/apt/lists/*
COPY . ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 8080
CMD ["python", "main.py"]