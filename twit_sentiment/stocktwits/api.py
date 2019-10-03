"""
Content
=================
Do not modify, edit or otherwise change the message content as passed through the API, 
except as needed to reformat for technical limitations of your specific service.
All $TICKER cashtags within each message must be hyperlinked and point to the Stocktwits ticker page at http://www.stocktwits.com/symbol/.
All @mentions within each message must be hyperlinked and point to the Stocktwits user page for the mentioned user at http://www.stocktwits.com/.
Messages should be presented with a timestamp, either in absolute (May 1, 2012 3:30pm) or relative (1 hour ago) format. 
The timestamp should be linked to the message display page on Stocktwits at http://www.stocktwits.com/message/.
If you choose to display media objects passed within the API in your application; such as a chart image or video, 
the object must be hyperlinked and point to the message display page on Stocktwits at http://www.stocktwits.com/message/.
Linking Cashtags (eg. $GOOG) can be done easily, we have created a library that provides text processing routines for Stocktwits Messages. 
This library provides autolinking and extraction for cashtags, it's available on Github: https://github.com/stocktwits/stocktwits-text-js

We recommend using the Twitter text-js Library for linking URLs. 
The library provides autolinking and extraction for URLs. You will find it on Github:  https://github.com/twitter/twitter-text-js

Author and Attribution
========================
The Stocktwits user’s name must be presented as the author of the content, 
using either the user’s Stocktwits user name or user name and full name. 
The name should be linked to the user’s Stocktwits profile page at http://www.stocktwits.com/.
The user’s name should be presented in a way to distinguish it from the message content.
Display of the user’s avatar is recommended, but not required. 
If displayed the user’s avatar should also be linked to the Stocktwits user page for the user.

Message Interactions
=======================
You may include Reply, Reshare, and Like action links using the Stocktwits web intents, 
if so they should be used consistently across all messages for authenticated Stocktwits users.
You cannot include third party interactions with Stocktwits messages.

Branding
========================
It must always be apparent to the user that they are looking at a Stocktwits message.
Any Stocktwits messages displayed individually or as part of a stream labeled with a Stocktwits logo 
adjacent to the message or stream. Logos and buttons are available via our logo page. Whenever possible, 
Stocktwits logos should link to http://stocktwits.com
"""

import config.configure
import logging as log
import datetime
from time import mktime, time
from requestors.requestors import ST_BASE_PARAMS, ST_BASE_URL


# Select which library to use for handling HTTP request.  If running on Google App Engine, use `GAE`.
# Otherwise, use `Requests` which is based on the `requests` module.
from requestors.requestors import Requests as R

RATE_LIMITS_HEADER = 'X-RateLimit-Remaining'
SERVER_DATE = 'Date'
RESET_LIMITS_AT_UNIX = 'X-RateLimit-Reset'
# Example list of exchanges to limit a watchlist to
EXCHANGES = ['NYSE', 'NASDAQ', 'NYSEMkt', 'NYSEArca']

# ---------------------------------------------------------------------
# Basic StockTwits interface
# ---------------------------------------------------------------------
def move_cursor(cursor=None):
    """ Moves the cursor through the API to get more info
        i.e. of @param cursor = { "more": true, "since": 10, "max": 12}
    """
    if cursor is None:
        raise 'No cursor to move'
    elif cursor['more'] is False:
        return None
    else:
        # update cursor to start at `since` and get maximum
        return dict(since=cursor['since'])

def watchlists():
    """ Gets user watch lists
    """
    wl = R.get_json(ST_BASE_URL + 'watchlists.json', params=ST_BASE_PARAMS)
    return wl['watchlists']
    
def get_trending_stocks(params={}, show_post=True):
    """ Get posts being streaming as trending
        PARAMS NOTE:
        `since`:	Returns results with an ID greater than (more recent than) the specified ID.
        `max`:	    Returns results with an ID less than (older than) or equal to the specified ID.
        `limit`:	Default and max limit is 30. This limit must be a number under 30.
        `callback`: Define your own callback function name, add this parameter as the value.
        `filter`:	Filter messages by links, charts, videos, or top. (Optional)
        if @params show_post = False
            Gets up to 30 symbols from trending stream
            NOTE: params -> `limit`: Default and max limit is 30. This limit must be a number under 30.
    """
    if show_post:
        all_params = ST_BASE_PARAMS.copy()
        try:
            for k, v in params.iteritems():
                all_params[k] = v
        except:
            pass
        try:
            wl = R.get_json(ST_BASE_URL + 'streams/trending.json', params=all_params)
        except BaseException as reset_time:
            r_at = reset_time.args[0]
            wait = float(r_at) - time()
            print('waiting for... ', wait, 'seconds')
            return
    else:
        try:
            wl = R.get_json(ST_BASE_URL + 'trending/symbols.json', params=all_params)
        except BaseException as reset_time:
            r_at = reset_time.args[0]
            wait = float(r_at) - time()
            print('waiting for... ', wait, 'seconds')
            return
        wl = dict(messages=wl['symbols'])# wl['watchlist']# ['symbols']
    return wl

def get_watched_stocks(wl_id, show_post=False, params={}):
    """ Get list of symbols being watched by specified StockTwits watchlist
        if @params show_post = True
            Gets up to 30 messages from Watchlist (wl_id) according to additional params
    """
    if show_post:
        all_params = ST_BASE_PARAMS.copy()
        try:
            for k, v in params.iteritems():
                all_params[k] = v
        except:
            pass
        wl = R.get_json(ST_BASE_URL + 'streams/watchlist/{}.json'.format(wl_id), params=all_params)
        wl = dict(cursor=wl['cursor'],messages=wl['messages'])
    else:
        wl = R.get_json(ST_BASE_URL + 'watchlists/show/{}.json'.format(wl_id), params=all_params)
        wl = dict(cursor=wl['cursor'],messages=wl['messages'])# wl['watchlist']# ['symbols']
    return wl

def get_stock_stream(symbol, params={}):
    """ gets stream of messages for given symbol
    """
    all_params = ST_BASE_PARAMS.copy()
    for k, v in params.iteritems():
        all_params[k] = v
    return R.get_json(ST_BASE_URL + 'streams/symbol/{}.json'.format(symbol), params=all_params)

def add_to_watchlist(symbols, wl_id):
    """ Adds list of symbols to our StockTwits watchlist.  Returns list of new symbols added
    """
    deadline = 30 * len(symbols)
    symbols = ','.join(symbols)  # must be a csv list
    params = ST_BASE_PARAMS.copy()
    params['symbols'] = symbols
    resp = R.post_json(ST_BASE_URL + 'watchlists/{}/symbols/create.json'.format(wl_id), params=params, deadline=deadline)
    if resp['response']['status'] == 200:
        return [s['symbol'] for s in resp['symbols']]
    else:
        return []

def delete_from_watchlist(symbol, wl_id):
    """ removes a single "symbols" (str) from watchlist.  Returns True on success, False on failure
    """
    params = ST_BASE_PARAMS.copy()
    params['symbols'] = symbol
    resp = R.post_json(ST_BASE_URL + 'watchlists/{}/symbols/destroy.json'.format(wl_id), params=params)
    if resp['response']['status'] == 200:
        return True
    else:
        return False


# def get_trending_stocks():
#     """ returns list of trending stock symbols, ensuring each symbol is part of a NYSE or NASDAQ
#     """
#     trending = R.get_json(ST_BASE_URL + 'trending/symbols.json', params=ST_BASE_PARAMS)['symbols']
#     symbols = [s['symbol'] for s in trending if s['exchange'] in EXCHANGES]
#     return symbols

def clean_watchlist(wl_id):
    """ Deletes stocks to follow if they aren't part of NASDAQ or NYSE
    """
    wl = R.get_json(ST_BASE_URL + 'watchlists/show/{}.json'.format(wl_id),
                  params=ST_BASE_PARAMS)['watchlist']['symbols']
    qty_deleted = 0
    for sym in wl:
        if sym['exchange'] not in EXCHANGES:
            log.info("Removing {}".format(sym))
            if delete_from_watchlist(sym['symbol'], wl_id=wl_id):
                qty_deleted += 1
            else:
                log.error("Error deleting symbol from watchlist: {}".format(sym))
    return qty_deleted