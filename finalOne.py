from flask import Flask, jsonify, render_template, request
app = Flask(__name__)
from tensorflow import keras

from keras.preprocessing.text import Tokenizer

models = keras.models.load_model("twitter_stress_detection.h5")

from tensorflow.keras.preprocessing.sequence import pad_sequences
MAX_SEQUENCE_LENGTH = 280
MAX_NB_WORDS = 25000

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)

    

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        def find(input):
            sequences_d = tokenizer.texts_to_sequences(input)
            data_d = pad_sequences(sequences_d, maxlen=MAX_SEQUENCE_LENGTH)
            k = models.predict(data_d)
            # s = models.predict(input)
            
            m = k[0][0]
            print("predicted",m)
            return {"m":m}
        print(request.form["name"])
        val = find(request.form["name"])
        print(val)
        return render_template('home.html', value=val)
    return render_template("home.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)