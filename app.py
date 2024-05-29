from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
from api_client import start_chat_and_send_message
import threading

app = Flask(__name__)

model_loaded = False
lock = threading.Lock()

model = None

@app.before_request
def load_model():
    global model, model_loaded
    if not model_loaded:
        with lock:
            if not model_loaded:
                model = joblib.load('./models/random_forest_model.pkl')
                model_loaded = True

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/diagnose', methods=['POST'])
def diagnose():
    global model_loaded

    if model is None:
        return jsonify({'error': 'Model không được tải. Vui lòng kiểm tra lại tệp mô hình.'})

    try:
        data = [
            float(request.form['pregnancies']),
            float(request.form['glucose']),
            float(request.form['blood_pressure']),
            float(request.form['skin_thickness']),
            float(request.form['insulin']),
            float(request.form['bmi']),
            float(request.form['dpf']),
            float(request.form['age'])
        ]

        data = np.array(data).reshape(1, -1)
        prediction = model.predict(data)
        prediction_proba = model.predict_proba(data)[0][1] * 100

        if prediction[0] == 1:
            diagnosis = "Bạn có nguy cơ cao mắc tiểu đường."
        else:
            diagnosis = "Sức khỏe tốt"

        response_data = {'diagnosis': diagnosis, 'accuracy': prediction_proba, 'data': data.tolist()}
        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': f'Có lỗi xảy ra: {str(e)}'})

@app.route('/get_advice', methods=['POST'])
def get_advice():
    try:
        request_data = request.get_json()
        data = request_data['data']

        input_parameters = f"Số lần mang thai: {data[0][0]}, Mức glucose: {data[0][1]}, Huyết áp: {data[0][2]}, Độ dày da: {data[0][3]}, Mức insulin: {data[0][4]}, Chỉ số BMI: {data[0][5]}, DPF: {data[0][6]}, Tuổi: {data[0][7]}"
        generative_response, _ = start_chat_and_send_message(f"hãy ra lời khuyên cải thiện sức khỏe dựa trên tỉ lệ mắc bệnh tiểu đường {request_data['accuracy']} Dựa trên các thông số: {input_parameters} lưu ý bỏ qua các câu mang nghĩa tương tự: 'Rất tiếc, tôi không thể đưa ra lời khuyên y tế'")
        
        return jsonify({'advice': generative_response})
    except Exception as e:
        return jsonify({'error': f'Có lỗi xảy ra trong quá trình tạo lời khuyên: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)
