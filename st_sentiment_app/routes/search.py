from flask import Blueprint, render_template, request, jsonify
import requests, json

STOCK_TWITS = 'https://stocktwits.com/symbol/'
search_term = ""
DEVEL_PROD = "localhost" # "elasticsearch"
ML_DEVEL_PROD = "localhost" # "mlapi"

headers = {
    'Content-Type': "application/json",
    'cache-control': "no-cache",
}

def first_symbol(of_hits):
    hits = of_hits['hits']
    for hit in hits:
        h = hit['_source']
        try:
            assert 'symbols' in h
            return h['symbols'][0]['symbol'], h['symbols'][0]['title']
        except AssertionError:
            print('there was an error finding symbol')
            pass
    return
    
            

def disect_hits(search,from_hits_key):
    """ Fn to collect twits by symbols
        Returns a dict w/ symbol as key and list of str as vals
    """
    # TODO: Add more relevancy
    rtd = {str(search): None}
    l=list()
    hits = from_hits_key['hits']
    for hit in hits:
        # sym = str(hit['_source']['source']['symbol'])
        bod = str(hit['_source']['body'])
        l.append(bod)
        # if sym in rtd:
        #     rtd[sym].append(bod)
        # else:
        #     rtd[sym] = list(bod)

    rtd[str(search)] = l
    return rtd

def constuct_mlapi_msg(disected_dic):
    """ constucts the message to ask ml for predictions
        Returns: json body ready to be OTW
    """
    all_msgs = list()
    for sym, ba in disected_dic.items():
        all_msgs.append(dict(SentimentText=ba, Sentiment=[0 for i in range(len(ba))]))
    return all_msgs


def ask_ml(msgs):
    """ makes a call out to service
        performs statistical calculations
        for front end
        Returns a dict of symbols as keys, tuples with percentages
    """
    
    url = "http://"+ML_DEVEL_PROD+":8765/prediction"
    for msg in msgs:
        payload = json.dumps(msg)
        response = requests.request("POST", url, data=payload, headers=headers)
        response_dict_data = json.loads(str(response.text))
        pos = 0
        neg = 0
        try:
            pos = sum(response_dict_data["Sentiment"])/float(len(response_dict_data["Sentiment"])) * 100.0
            neg = 100.0 - pos
        except TypeError:
            pass
        finally:
            response_dict_data['Sentiment'] = [pos, neg]
            return response_dict_data


# creating a Blueprint class
search_blueprint = Blueprint('search',__name__,template_folder="templates")
@search_blueprint.route("/",methods=['GET','POST'],endpoint='index')
def index():
    if request.method=='GET':
        res ={
	            'hits': {'total': 0, 'hits': []},
                'symbol': "",
                'Sentiment': [0,0],
                'company': "",
                'done': 0
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
            # print(response_dict_data)
            if response_dict_data['hits']['total'] is 0:
                return render_template('err.html',err={'code':404,'msg':'Not Found.'})
            dh = disect_hits(search_term, response_dict_data['hits'])
            msgs = constuct_mlapi_msg(dh)
            response_from_mlapi = ask_ml(msgs)
            fs, fs1 = first_symbol(response_dict_data['hits'])
            print(fs, fs1)
            response_from_mlapi['symbol'] = '$'+fs
            response_from_mlapi['company'] = fs1
            response_from_mlapi['link'] = STOCK_TWITS+fs
            response_from_mlapi['done'] = 1
            # print('*'*100)
            print("reply is ~>", response_from_mlapi)
            # print('*'*100)
            return render_template('index.html', res=response_from_mlapi)


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
        return json.dumps(response_dict_data)



