from flask import Blueprint, render_template, request, jsonify
import requests, json

# creating a Blueprint class
search_blueprint = Blueprint('search',__name__,template_folder="templates")
search_term = ""
DEVEL_PROD = "localhost" # "elasticsearch"

headers = {
    'Content-Type': "application/json",
    'cache-control': "no-cache",
}

@search_blueprint.route("/",methods=['GET','POST'],endpoint='index')
def index():
    if request.method=='GET':
        res ={
	            'hits': {'total': 0, 'hits': []}
        }
        return render_template("index.html",res=res)
    elif request.method =='POST':
        if request.method == 'POST':
            print("-----------------Calling search Result----------")
            search_term = request.form["input"]
            print("Search Term:", search_term)
            payload = {
                "query": {
                    "query_string": {
                        "analyze_wildcard": True,
                        "query": str(search_term),
                        # "fields": ["topic", "title", "url", "labels"]
                        "fields": ["body"]
                    }
                    # ,
                    # "nested": {
                    #     "path": "symbols",
                    #     "query": {
                    #         "bool": {
                    #         "must": [
                    #                 { "match": { "symbols.symbol": str(search_term) }},
                    #                 { "match": { "symbols.title":  str(search_term) }} 
                    #             ]
                    #         }
                    #     },
                    #     "inner_hits": { 
                    #         "highlight": {
                    #             "fields": {
                    #                 "symbols.symbol": {}
                    #             }
                    #         }
                    #     }
                    # }
                },
                "size": 50,
                "sort": [

                ]
            }
            payload = json.dumps(payload)
            url = "http://"+DEVEL_PROD+":9200/stocktwits/twits/_search"
            response = requests.request("GET", url, data=payload, headers=headers)
            response_dict_data = json.loads(str(response.text))
            print(response_dict_data)
            return render_template('index.html', res=response_dict_data)


@search_blueprint.route("/autocomplete",methods=['POST'],endpoint='autocomplete')
def autocomplete():
    if request.method == 'POST':
        search_term = request.form["input"]
        print("POST request called")
        print(search_term)
        payload ={
          "autocomplete" : {
            "text" : str(search_term),
            "completion" : {
              "field" : "title_suggest",
              "field": "symbol_suggest"
            }
          }
        }
        payload = json.dumps(payload)
        url="http://"+ DEVEL_PROD +":9200/autocomplete/_suggest"
        response = requests.request("GET", url, data=payload, headers=headers)
        response_dict_data = json.loads(str(response.text))
        # print(response_dict_data)
        return json.dumps(response_dict_data)



