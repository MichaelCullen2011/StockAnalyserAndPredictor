# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.9-slim

# ARG ssh_prv_key
# ARG ssh_pub_key

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True
# RUN apt-get update && \
#     apt-get install -y \
#         git \
#         openssh-server \
#         default-libmysqlclient-dev

# # Authorize SSH Host
# RUN mkdir -p /root/.ssh && \
#     chmod 0700 /root/.ssh && \
#     ssh-keyscan github.com > /root/.ssh/known_hosts

# # Add the keys and set permissions
# RUN echo "$ssh_prv_key" > /root/.ssh/id_ed25519 && \
#     echo "$ssh_pub_key" > /root/.ssh/id_ed25519.pub && \
#     chmod 600 /root/.ssh/id_ed25519 && \
#     chmod 600 /root/.ssh/id_ed25519.pub

# Copy local code to the container image.
WORKDIR /app
COPY requirements.txt .


# Install production dependencies.
# RUN apt update && apt install -y git
RUN pip install -r requirements.txt

COPY . .

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
ENV FLASK_APP=src/app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=development

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]