FROM python:3.12.3

WORKDIR /back-end

COPY ./requirements.txt /back-end/requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 

RUN pip install --no-cache-dir --upgrade -r /back-end/requirements.txt

COPY ./app /back-end/app

CMD ["fastapi", "run", "app/main.py", "--port", "80"]