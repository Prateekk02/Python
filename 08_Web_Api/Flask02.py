from flask import Flask,render_template, request, jsonify



app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def home_page():
    return render_template('index.html')   # Automatically render the index.html file.


@app.route('/math', methods  = ['POST'])
def math_operation():

    if(request.method == 'POST'):

        ops =   request.form['operation']
        num1 = int(request.form['num1'])
        num2 = int(request.form['num2'])

        if(ops == 'add'):
            res = num1+num2
            result = 'The sum of '+str(num1) + ' and ' +str(num2) + ' is ' + str(res)
        # return render_template('results.html', result = result)

        if(ops == 'subtract'):
            res = num1-num2
            result = 'The subtraction of '+str(num1) + ' and ' +str(num2) + ' is ' + str(res)
        # return render_template('results.html', result = result)

        if(ops == 'multiply'):
            res = num1*num2
            result = 'The multiplication of '+str(num1) + ' and ' +str(num2) + ' is ' + str(res)
        # return render_template('results.html', result = result)
        try:
            if(ops == 'divide'):
                res = num1/num2
                result = 'The division of '+str(num1) + ' and ' +str(num2) + ' is ' + str(res)
            return render_template('results.html', result = result)
        except ZeroDivisionError as e:
            return render_template('results.html',result = str(e))


# Passing data through postman

@app.route('/postman_data', methods  = ['POST'])
def math_operation1():

    if(request.method == 'POST'):

        ops =   request.json['operation']
        num1 = int(request.json['num1'])
        num2 = int(request.json['num2'])

        if(ops == 'add'):
            res = num1+num2
            result = 'The sum of '+str(num1) + ' and ' +str(num2) + ' is ' + str(res)
        # return render_template('results.html', result = result)

        if(ops == 'subtract'):
            res = num1-num2
            result = 'The subtraction of '+str(num1) + ' and ' +str(num2) + ' is ' + str(res)
        # return render_template('results.html', result = result)

        if(ops == 'multiply'):
            res = num1*num2
            result = 'The multiplication of '+str(num1) + ' and ' +str(num2) + ' is ' + str(res)
        # return render_template('results.html', result = result)
        try:
            if(ops == 'divide'):
                res = num1/num2
                result = 'The division of '+str(num1) + ' and ' +str(num2) + ' is ' + str(res)
        except ZeroDivisionError as e:
            result = str(e)
        return jsonify(result)





if __name__ == "__main__":
    app.run(host = "0.0.0.0")