from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 8)  # Thay đổi kích thước tùy thuộc vào số thuộc tính
    loaded_model = pickle.load(open("model.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = []
        fields = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'diabetes_pedigree_function', 'age']
        
        for field in fields:
            if field in request.form:
                to_predict_list.append(request.form[field])
            else:
                # Nếu một trường không tồn tại, có thể thực hiện xử lý khác tùy thuộc vào yêu cầu của bạn.
                # Ở đây, tôi thêm giá trị 0 cho trường không tồn tại.
                to_predict_list.append(0)

        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)

        if int(result) == 1:
            prediction = 'Người này không mắc bệnh tiểu đường'
            
        else:
            prediction = 'Người này mắc bệnh tiểu đường'
            
        return render_template("result.html", prediction=prediction)
        
if __name__ == '__main__':
    app.run(debug=True)
