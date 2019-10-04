from stocktwits import api as ST
import json
import time



def runner(i=None):
    e_i = -1
    try:
        res = ST.get_trending_stocks()
        combined_results = res['messages']
        js = json.dumps(combined_results, sort_keys=True, indent=5)
        f = open("dataSample.json", "a+")
        f.write(js)
        for i in range(100000):
            # move cursor
            print('at iteration ', i+1)
            new_cur = ST.move_cursor(res['cursor'])
            time.sleep(20)
            res = ST.get_trending_stocks(new_cur)
            if res is None:
                return i
            else:
                combined_results = res['messages']
                js = json.dumps(combined_results, sort_keys=True, indent=5)
                f.write(js)
            e_i = i
    except Exception as e:
        print(e.with_traceback())
        f.close()
        return e_i
    f.close()





if __name__ == "__main__":
    i = None
    while True:
        i = runner(i)
        if i is None:
            break