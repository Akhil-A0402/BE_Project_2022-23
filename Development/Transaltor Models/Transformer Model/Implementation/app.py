from flask import Flask,render_template, request
import tensorflow as tf
import os
import string
import re
import pickle
import Transformer

strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

transformer = Transformer.load_pretrained()
print("Model loaded ok")
eng_vector, hin_vector = Transformer.textVectorization()
print("vectors loaded ok")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/", methods=["POST"])
def predict():
    form_keys = request.values
    eng = form_keys["eng"]
    hin = ""
    try:
        hin = Transformer.decode_sequence(eng, hin_vector, eng_vector, transformer)
    except:
        hin = "Error"
    return render_template("home.html", input_text=eng ,prediction_text=hin)

if __name__ == "__main__":
    app.run(debug=True)