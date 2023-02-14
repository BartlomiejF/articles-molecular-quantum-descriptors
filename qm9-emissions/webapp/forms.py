from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

class ChromophoreForm(FlaskForm):
    chromophore = StringField("Chromophore SMILES", validators=[DataRequired()])
    # solvent = StringField("Solvent SMILES")
    submit = SubmitField('Predict')