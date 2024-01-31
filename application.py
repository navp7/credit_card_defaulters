from flask import Flask, request, render_template, jsonify, send_file
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import os,sys
#from src.pipeline.training_pipeline import TrainPipeline
from src.pipeline.prediction_pipeline import PredictionPipeline

application = Flask(__name__)

app=application

@app.route('/')
def home_page():
    return  render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def upload():
    
    try:
        if request.method == 'POST':
            # it is a object of prediction pipeline
            prediction_pipeline = PredictionPipeline(request)
            
            #now we are running this run pipeline method
            prediction_file_detail = prediction_pipeline.run_predict_pipeline()

            logging.info("Prediction completed. Downloading prediction file.")
            print("Prediction completed. Downloading prediction file.")
            return send_file(prediction_file_detail.prediction_file_path,
                            download_name= prediction_file_detail.prediction_file_name,
                            as_attachment= True)

        else:
            return render_template('upload2.html')
    except Exception as e:
        raise CustomException(e,sys)
    

# code begins from here

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug= True)