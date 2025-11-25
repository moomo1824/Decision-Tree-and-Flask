import pickle
from flask import Flask, render_template, request

app = Flask(__name__, template_folder="template")

model = pickle.load(open("model/titanic.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def pred():
    pc = int(request.form.get("pclasss"))
    se = int(request.form.get("sexx"))
    ag = float(request.form.get("agee"))
    si = int(request.form.get("sibspp"))
    pa = int(request.form.get("parchh"))
    fa = float(request.form.get("faree"))
    cl = int(request.form.get("classs"))
    wh = int(request.form.get("whoo"))
    am = int(request.form.get("adult_malee"))
    de = int(request.form.get("deckk"))
    et = int(request.form.get("embark_townn"))
    al = int(request.form.get("alivee"))
    alo = int(request.form.get("alonee"))

    feature = [[pc, se, ag, si, pa, fa, cl, wh, am, de, et, al, alo]]
    prediction = model.predict(feature)
    output = prediction[0]

    if output == 0:
        result = "The person is Dead"
    else:
        result = "The person survived"

    return render_template("index.html", predicted_text=f'Prediction is {output}: {result}')

if __name__ == "__main__":
    app.run(debug=True)
