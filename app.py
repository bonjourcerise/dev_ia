from flask import Flask, request, render_template
import pandas as pd
import joblib
import logging
from opencensus.ext.azure.log_exporter import AzureLogHandler

# Declare a Flask app
app = Flask(__name__)

# Azure Monitor
logger = logging.getLogger(__name__)
logger.addHandler(AzureLogHandler(
    connection_string='InstrumentationKey=#')
)

# Main function here
@app.route('/', methods=['GET', 'POST'])
def main():
    
    # If a form is submitted
    if request.method == "POST":
        
        # Unpickle classifier
        model = joblib.load("model.pkl")
        
        # Get values through input bars
        concave_points_worst = request.form.get("concave_points_worst")
        concave_points_mean = request.form.get("concave_points_mean")
        radius_worst = request.form.get("radius_worst")
        perimeter_worst = request.form.get("perimeter_worst")
        compactness_worst = request.form.get("concave_points_mean")
        symmetry_worst = request.form.get("symmetry_worst")
        texture_worst = request.form.get("texture_worst")
        area_se = request.form.get("area_se")
        concavity_mean = request.form.get("concavity_mean")
        area_worst = request.form.get("area_worst")
        texture_mean = request.form.get("texture_mean")
        
        # Put inputs to dataframe
        X = pd.DataFrame([[concave_points_worst, concave_points_mean, radius_worst,perimeter_worst,compactness_worst,symmetry_worst,texture_worst,area_se,concavity_mean,area_worst,texture_mean]],
        columns = ["concave points_worst", "concave points_mean", "radius_worst","perimeter_worst","compactness_worst","symmetry_worst","texture_worst","area_se","concavity_mean","area_worst","texture_mean"])
        
        # Get prediction
        prediction = model.predict(X)[0]
        probability_0 = model.predict_proba(X)[:,0]
        probability_1 = model.predict_proba(X)[:,1]

        # Register prediction in Azure Monitor
        logger.warning(prediction)

        if prediction == 0:
            prediction_text = "La patiente est diagnostiquée avec une tumeur bénigne."
            probability_text = "Indice de confiance (/100): {!s:5.5}]".format(probability_0*100)
        else:
            prediction_text = "La patiente est diagnostiquée avec une tumeur maligne."
            probability_text = "Indice de confiance (/ 100) ): {!s:5.5}]".format(probability_1*100)
        
        return render_template("index.html", output1 = prediction_text, output2 = probability_text)
    
    else:
        prediction = ""

    return render_template("index.html")

# Running the app
if __name__ == '__main__':
    app.run(debug = True)