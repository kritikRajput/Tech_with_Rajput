from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
from twilio.rest import Client
from flask_cors import CORS



app = Flask(__name__)
CORS(app)  

model = joblib.load('models/fraud_model.pkl')

scaler = joblib.load('models/scaler.pkl')
pca = joblib.load('models/pca.pkl')


account_sid = 'AC75aef13b86cc18cbc40e7801b4595cde'
auth_token = 'eda1be29301f2c190811349054ed59f7'
twilio_client = Client(account_sid, auth_token)

@app.route('/')
def home():
    return render_template('index.html')  
  


@app.route('/detect_fraud', methods=['POST'])
def detect_fraud():
    
    try:
       
        data = request.get_json()
        print(f"Data received: {data}")
        transaction_time = float(data['transaction_time'])
        transaction_amount = float(data['transaction_amount'])

       

        input_data = pd.DataFrame([[transaction_time, transaction_amount]], columns=['Time', 'Amount'])
        print(f"Input data for prediction: {input_data}")
        scaled_data = scaler.transform(input_data)
        print(f"Scaled data: {scaled_data}")

        pca_data = pca.transform(scaled_data)
        print(f"PCA transformed data: {pca_data}")

        prediction = model.predict(pca_data)
        print(f"Prediction: {prediction}")

       
      
        if prediction[0] == 1:
            
            send_sms_alert(transaction_time, transaction_amount, fraud_detected=True)
            print("Fraud detected, SMS alert sent.")
            return jsonify({'fraud': True}) 
        else:
            
            send_sms_alert(transaction_time, transaction_amount, fraud_detected=False)
            print("Legitimate transaction, SMS sent.")
            return jsonify({'fraud': False})  

    except Exception as e:
        return jsonify({'error': str(e)})


def send_sms_alert(transaction_time, transaction_amount, fraud_detected):
    try:
        if fraud_detected:
            message_body = f"Fraud Alert! Suspicious transaction detected. Time: {transaction_time}, Amount: {transaction_amount}"
        else:
            message_body = f"Transaction Approved. Time: {transaction_time}, Amount: {transaction_amount}."

        message = twilio_client.messages.create(
            body=message_body,
            from_='+13204274255',  
            to='+919917052327'      
        )
        print(f'SMS sent: {message.sid}')
    except Exception as e:
        print(f'Failed to send SMS: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
