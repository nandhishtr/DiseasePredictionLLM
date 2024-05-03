from flask import Flask, render_template, request, jsonify
from med_bot import predict

app = Flask(__name__)


# Home route
@app.route('/')
def home():
    return render_template('index.html')


# Route to handle user input and return bot's response
@app.route('/send-message', methods=['POST'])
def send_message():
    user_input = request.form['user_input']
    response = predict(user_input)
    return jsonify({'bot_response': response})


if __name__ == '__main__':
    app.run(port=3000, debug=True)
