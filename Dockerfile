FROM python:3.11-slim
ENV PYTHONBUFFERED True
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 8080
CMD ["python", "main.py", "--host=0.0.0.0", "--port=8080"]