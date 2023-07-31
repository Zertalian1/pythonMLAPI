from flask import Flask, jsonify, request
from flask_restful import Resource, Api
from json import loads

from PredictProductsCount import PredictProductsCount
from PredictProductsPrice import PredictProductsPrice

app = Flask(__name__)
api = Api(app)


class ApiInfo(Resource):

    def get(self):
        api_info = open('apiInfo/apiInfo.txt', 'r', encoding='utf-8')
        return jsonify({'message': api_info.readlines()})


class PredictPrice(Resource):

    def get(self):
        api_info = open('apiInfo/predictPrice.txt', 'r', encoding='utf-8')
        return jsonify({'message': api_info.readlines()})

    def post(self):
        year = request.get_json()['year']
        print(year)
        model_service = PredictProductsPrice()
        result = model_service.predict_results(year)
        parsed = loads(result.to_json(orient="split"))
        return parsed


class PredictFood(Resource):

    def get(self):
        api_info = open('apiInfo/predictFood.txt', 'r', encoding='utf-8')
        return jsonify({'message': api_info.readlines()})

    def post(self):
        params = loads(request.data)
        year = None
        peoples = None
        products = None
        if 'year' in params:
            year = request.get_json()['year']
        if 'peoples' in params:
            peoples = request.get_json()['peoples']
        if 'products' in params:
            products = list(map(int, request.get_json()['products']))
        model_service = PredictProductsCount()
        if products is None:
            result = model_service.predict_results_all_prod(year, peoples)
        else:
            result = model_service.predict_results_selected_prod(products, year, peoples)
        parsed = loads(result.to_json(orient="split"))
        return parsed


api.add_resource(ApiInfo, '/')
api.add_resource(PredictPrice, '/predict-price')
api.add_resource(PredictFood, '/predict-food')

if __name__ == '__main__':
    app.run(debug=True)
