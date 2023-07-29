from flask import Flask, jsonify, request
from flask_restful import Resource, Api
from json import loads, dumps

from PredictProductsCount import PredictProductsCount
from PredictProductsPrice import PredictProductsPrice

app = Flask(__name__)
api = Api(app)


class Hello(Resource):

    def get(self):
        api_info = open('apiInfo.txt', 'r')
        return jsonify({'message': api_info.readlines()})

    # Corresponds to POST request
    def post(self):
        data = request.get_json()  # status code
        print(data)
        return data


class PredictPrice(Resource):

    def post(self):
        year = request.get_json()['year']
        print(year)
        model_service = PredictProductsPrice()
        result = model_service.predict_results(year)
        parsed = loads(result.to_json(orient="split"))
        return parsed


class PredictFood(Resource):

    def post(self):
        year = request.get_json()['year']
        peoples = request.get_json()['peoples']
        model_service = PredictProductsCount()
        result = model_service.predict_results(year, peoples)
        parsed = loads(result.to_json(orient="split"))
        return parsed


api.add_resource(Hello, '/')
api.add_resource(PredictPrice, '/predict-price')
api.add_resource(PredictFood, '/predict-food')

# driver function
if __name__ == '__main__':
    app.run(debug=True)
