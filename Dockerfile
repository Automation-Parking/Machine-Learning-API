FROM python:3.11-slim
ENV PYTHONBUFFERED True
ENV APP_HOME /app
WORKDIR $APP_HOME
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libopencv-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*
COPY . ./
RUN mkdir -p image_uploads/object-detect/images/ image_uploads/OCR/images/
RUN python -c "import os; print(os.listdir('image_uploads/object-detect/'))"
RUN python -c "import os; print(os.listdir('image_uploads/OCR/'))"
RUN python -c "import os; print(os.listdir('model/'))"
RUN pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir
EXPOSE 8080
CMD ["python", "main.py"]