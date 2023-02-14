from flask import Flask, render_template, request
import predictor
import forms

app = Flask(__name__)
app.config['SECRET_KEY'] = 'you-will-never-guess'


@app.route("/", methods=["GET", "POST"])
def main():
    form = forms.ChromophoreForm()
    if request.method == 'POST':
        smiles = f"{form.chromophore.data}".strip()
        return predictor.predict(smiles)
    return render_template("main.html", form=form)


@app.route("/<smiles>", methods=["GET"])
def predict(smiles):
    return predictor.predict(smiles)


if __name__ == "__main__":
    app.run(
        debug=True,
        )
