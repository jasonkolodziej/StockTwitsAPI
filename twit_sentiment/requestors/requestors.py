""" A variety of classes using different libraries to implement `get_json` and `post_json` methods.
"""
import os
import logging as log
import urllib
import json
# Try to import modules needed for Google App Engine just in case
try:
    from google.appengine.api import urlfetch
    from google.appengine.runtime import DeadlineExceededError
except:
    pass

# Try to import requests just in case
try:
    import requests
except:
    pass


# StockTwits details
ST_BASE_URL = 'https://api.stocktwits.com/api/2/'
ST_BASE_PARAMS = dict(access_token=os.getenv('ST_ACCESS_TOKEN'))
print(ST_BASE_PARAMS)


class Requests():
    """ Uses `requests` library to GET and POST to Stocktwits, and also to convert responses to JSON
    """
    @staticmethod
    def get_json(url, params=None):
        """ Uses tries to GET a few times before giving up if a timeout.  returns JSON
        """
        resp = None
        for i in range(4):
            try:
                resp = requests.get(url, params=params, timeout=5)
            except requests.Timeout:
                trimmed_params = {k: v for k, v in params.iteritems() if k not in ST_BASE_PARAMS.keys()}
                log.error('GET Timeout to {} w/ {}'.format(url[len(ST_BASE_URL):], trimmed_params))
            if resp is not None:
                break
        if resp is None:
            log.error('GET loop Timeout')
        # for StockTwits API
        try:
            if int(resp.headers['X-RateLimit-Remaining']) < 5:
                raise BaseException(int(resp.headers['X-RateLimit-Reset']))
            if resp.status_code is 429 or resp.status_code is None:
                raise BaseException(int(resp.headers['X-RateLimit-Reset']))
        except:
            raise
        return json.loads(resp.content)

    @staticmethod
    def post_json(url, params=None, deadline=30):
        """ Tries to post a couple times in a loop before giving up if a timeout.
        """
        resp = None
        for i in range(4):
            try:
                resp = requests.post(url, params=params, timeout=5)
            except requests.Timeout:
                trimmed_params = {k: v for k, v in params.iteritems() if k not in ST_BASE_PARAMS.keys()}
                log.error('POST Timeout to {} w/ {}'.format(url[len(ST_BASE_URL):], trimmed_params))
            if resp is not None:
                break
        # TODO wrap in appropriate try/except
        return json.loads(resp.content)

# TODO: add response 429 handle to exit
class GAE():
    """ A wrapper around Google App Engine's `urlfetch` to make it act like `requests` package
    """
    @staticmethod
    def get_json(url, params=None):
        """ Uses tries to GET a few times before giving up if a timeout.  returns JSON
        """
        params = ('?' + urllib.parse.urlencode(params)) if params else ''  # URL query string parameters (Access Token, etc)
        resp = None
        for i in range(4):
            try:
                resp = urlfetch.fetch(url+params, method='GET')
            except:
                trimmed_params = {k: v for k, v in params.iteritems() if k not in ST_BASE_PARAMS.keys()}
                log.error('GET Timeout to {} w/ {}'.format(url[len(ST_BASE_URL):], trimmed_params))
            if resp is not None:
                break
        if resp is None:
            log.error('GET loop Timeout')
            return None
        else:
            return json.loads(resp.content)
    @staticmethod
    def post_json(url, params=None, deadline=30):
        """ Tries to post a couple times in a loop before giving up if a timeout.
        """
        params = '?' + urllib.parse.urlencode(params) if params else ''  # URL query string parameters (Access Token)
        resp = None
        for i in range(4):
            try:
                resp = urlfetch.fetch(url+params, method='POST', deadline=deadline)
            except DeadlineExceededError:
                trimmed_params = {k: v for k, v in params.iteritems() if k not in ST_BASE_PARAMS.keys()}
                log.error('POST Timeout to {} w/ {}'.format(url[len(ST_BASE_URL):], trimmed_params))
            if resp is not None:
                break
        # TODO wrap in appropriate try/except
        return json.loads(resp.content)