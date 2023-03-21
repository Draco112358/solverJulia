from flask import Flask, jsonify, make_response, request
from flask_cors import CORS
from solver_py.solver import doSolving
from mesher_py.mesher import doMeshing

app = Flask(__name__)
CORS(app)

@app.route("/")
def hello_from_root():
    return jsonify(message='Hello from root!')


@app.route("/hello")
def hello():
    return jsonify(message='Hello from path!')
    

@app.route("/meshing", methods=['POST'])
def meshing():
    inputMesher = request.get_json()
    return doMeshing(inputMesher)


@app.route("/solving", methods=['POST'])
def solving():
    mesherOutput = request.get_json()['mesherOutput']
    solverInput = request.get_json()['solverInput']
    solverAlgoParams = request.get_json()['solverAlgoParams']
    return doSolving(mesherOutput, solverInput, solverAlgoParams)


@app.errorhandler(404)
def resource_not_found(e):
    return make_response(jsonify(error='Not found!'), 404)
