from flask import Flask
from flask import request
app = Flask(__name__)

@app.route("/")
def hello_world1():
    return "<h1>Hello, World!</h1>"

@app.route("/Grevious")
def hello_world2():
    return "<h1> Hello, there! </h1>"

@app.route("/Luke")
def hello_world3():
    return "<h1> Well,Hello, there! </h1>"
@app.route("/test")
def test():
    a = 5+6
    return "This is my funtion to run app {} ".format(a)

@app.route("/test2")    
def test2():
    data = request.args.get('x')      #  This will take argument
    return "This is my function to run app {} ".format(data)


if __name__=="__main__":
    app.run(host="0.0.0.0")
