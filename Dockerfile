FROM python:3.11-slim

# Common libraries strategies might want
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    ta

WORKDIR /strategy

CMD ["python", "worker.py"]
