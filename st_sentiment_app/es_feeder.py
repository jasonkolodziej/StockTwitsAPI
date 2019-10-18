import requests
import json

def create_auto_complete_payload(twit_payload):
    title_suggest = title = twit_payload["symbols"][0]["title"]
    symbol_suggest = symbol = twit_payload["symbols"][0]["symbol"]
    return {"title": title, "title_suggest": title_suggest, "symbol":symbol, "symbol_suggest":symbol_suggest}

if __name__ == "__main__":
    post_url = "http://localhost:9200/stocktwits/twits"
    post_autocomplete_url ="http://localhost:9200/autocomplete/titles"
    headers = {
    'Content-Type': "application/json",
    'cache-control': "no-cache"
    }
    count = 1
    with open('../twit_sentiment/samples/dataSample_merge_minify.json') as json_file:
        data = json.load(json_file)
        for p in data['messages']:
            payload = p
            payload_autocomplete = create_auto_complete_payload(p)
            payload = json.dumps(payload)
            payload_autocomplete = json.dumps(payload_autocomplete)
            response = requests.request("POST", post_url, data=payload, headers=headers)
            response_autocomplete = requests.request("POST", post_autocomplete_url, data=payload_autocomplete, headers=headers)
            if(response.status_code==201):
                print("Values Posted in stocktwits index")
            if(response_autocomplete.status_code==201):
                print("Values Posted in autocomplete index")
            print("----------------", count, "----------------------")
            count = count + 1