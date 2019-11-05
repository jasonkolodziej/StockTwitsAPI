import random
import csv
if __name__ == "__main__":
    twitter_data_path = 'training_data/twitter1_6m.csv'
    fin_data_path = 'training_data/tweet_sentiment.csv'
    shuffled_ts = 'training_data/full_shuffled_ts.csv'
    #combine the data
    with open(twitter_data_path,"r") as source:
        rdr = csv.reader( source )
        with open("result.csv","a+") as result:
            wtr = csv.writer( result )
            for r in rdr:
                # tweets from twitter
                wtr.writerow( (r[5], r[0]) )
                # financial tweets
                # cleaned_tweets,sentiment[0-4]
                if int(r[1]) == 0:
                    wtr.writerow((r[0], 2))
                elif int(r[1]) == -1:
                    wtr.writerow((r[0], 0))
                elif int(r[1]) == 1:
                    wtr.writerow((r[0], 4))
                else:
                    wtr.writerow((r[0], r[1]))
