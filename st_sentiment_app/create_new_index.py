import requests
import json

def check_if_index_is_present(url):
    response = requests.request("GET", url, data="")
    json_data = json.loads(response.text)
    return json_data


if __name__ == "__main__":
    url = "http://localhost:9200/_template/search_engine_template/"
    response = requests.request("GET", url, data="")
    if(len(response.text)>2):
        print("1. Deleted template: search_engine_template")
        response_delete = requests.request("DELETE", url)
    payload = {
            # site we scraped
          "template": "stocktwits",
          "settings": {
            "number_of_shards": 1
          },
          "mappings": {
            "twits":{
                "_source": {
                    "enabled": True
                },
                "properties":{
                    # posting data model from `www.stocktwits.com`
                    # @ref es_model.json
                    "body": {"type":"text"},
                    "created_at": {"type":"date"},
                    "entities": {
                        "properties":{
                            "chart":{
                                "properties":{
                                    "large" : {"type": "text"},
                                    "original": {"type": "text"},
                                    "thumb":{"type":"text"},
                                    "url":{"type":"text"}
                                }
                            },
                            "sentiment": {
                                "properties":{
                                    "basic": {"type":"keyword"}
                                } # // look for type
                            }
                        },
                        "id": {"type":"long"},
                        "liked_by_self": {"type": "boolean"},
                        "mentioned_users": {"type": "nested"},
                        "reshared_by_self": {"type": "boolean"},
                        "source": {
                            "properties":{
                                "id": {"type":"long"},
                                "title": {"type": "keyword"},
                                "url": {"type": "keyword"}
                            }
                        },
                        "symbols": {"type": "nested"},
                        "user": {
                            "properties":{
                                "avatar_url": {"type": "text"},
                                "avatar_url_ssl": {"type": "text"},
                                "classification": {"type": "nested"},
                                "followers": {"type": "long"},
                                "following": {"type": "long"},
                                "id": {"type": "long"},
                                "ideas": {"type": "text"},
                                "identity": {"type": "keyword"},
                                "join_date": {"type": "date"},
                                "like_count": {"type": "long"},
                                "name": {"type": "keyword"},
                                "official": {"type": "boolean"},
                                "plus_tier": {"type": "text"},
                                "premium_room": {"type": "text"},
                                "username": {"type": "keyword"},
                                "watchlist_stocks_count": {"type": "long"}
                            }
                        }
                    }
                }
            }
        }
    }
    payload = json.dumps(payload)
    headers = {
            'Content-Type': "application/json",
            'cache-control': "no-cache"
        }
    response = requests.request("PUT", url, data=payload, headers=headers)
    if (response.status_code == 200):
        print("2. Created a new template: search_engine_template")

    url = "http://localhost:9200/stocktwits"
    json_data = check_if_index_is_present(url)

    if(not 'error' in json_data):
        print("3. Deleted an index: stocktwits")
        response = requests.request("DELETE", url)

    response = requests.request("PUT", url)
    if (response.status_code == 200):
        print("4. Created an index: stocktwits")

    url = "http://localhost:9200/autocomplete"
    json_data = check_if_index_is_present(url)

    if(not 'error' in json_data):
        print("5. Deleting index: autocomplete")
        response = requests.request("DELETE", url)

    payload = {
      "mappings": {
        "titles" : {
          "properties" : {
            "title" : { "type" : "string" },
            "title_suggest" : {
              "type" :     "completion",
              "analyzer" :  "standard",
              "search_analyzer" : "standard",
              "preserve_position_increments": False,
              "preserve_separators": False
            },
            "symbol" : { "type" : "keyword" },
            "symbol_suggest" : {
              "type" :     "completion",
              "analyzer" :  "standard",
              "search_analyzer" : "standard",
              "preserve_position_increments": False,
              "preserve_separators": False
            }
          }
        }
      }
    }
    payload = json.dumps(payload)
    response = requests.request("PUT", url, data=payload, headers=headers)

    if(response.status_code==200):
        print("6. Created a new index: autocomplete")