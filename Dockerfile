FROM ubuntu:latest As dev
RUN apt update
RUN apt install -y python3 python3-distutils python3-dev python3-pip
RUN mkdir /stress
WORKDIR /stress
RUN pip3 install matplotlib nltk ftfy pandas scikit-learn tensorflow keras flask gensim
COPY app.py .
COPY GoogleNews-vectors-negative300.bin.gz .
COPY Sentiment_Analysis_Dataset_2.csv .
COPY vader_processed_final.csv .
COPY templates/ /stress/templates/
EXPOSE 5000
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
