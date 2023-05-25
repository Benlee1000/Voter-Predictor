from flask import Flask, render_template, request
import nonvoters as nv

app = Flask('nonvoters')

@app.route('/')
def show_predict_form():
    nonvoters = nv.Nonvoters()
    model_params, model = nonvoters.find_model() 
    predicted = nonvoters.find_predictions(model)
    image_buffer = nonvoters.save_model()
    accuracy, variance, mse, precision, recall = nonvoters.find_stats()
    return render_template('predictorform.html', predicted=predicted, buffer_encode=image_buffer, model_params=model_params, accuracy=accuracy, variance=variance, mse=mse, precision=precision, recall=recall)

@app.route('/results', methods=['POST'])
def show_result_form():
    if request.method == 'POST':
        education = request.form['education']
        race = request.form['race']
        gender = request.form['gender']
        income = request.form['income']

        nonvoters = nv.Nonvoters()
        _, model = nonvoters.find_model() 

        prediction = nonvoters.make_prediction(education, race, gender, income, model)

        return render_template('resultsform.html', model_output=prediction, education=education, race=race, gender=gender, income=income)

app.run("localhost", "9999", debug=True) 