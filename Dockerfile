FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY First_Data_Analysis.py cbb_analysis.py ./
COPY archive/cbb.csv archive/cbb.csv

CMD ["python", "First_Data_Analysis.py"]
