from custom_ml import ConcatPoolingGRUAdaptive, SentimentDataset, collate_fn
import json
vocab_size = 100000
model_vocab_size = vocab_size + 2
embedding_dim = 100
rnn_hidden = 256
n_out = 2




test_d = dict(SentimentText=["It made me cry.","It was so good it made me cry.","It's ok.","This movie had the best acting and the dialogue was so good. I loved it.","Garbage"], Sentiment=[0,0,0,0,0])



def run_prediction_on_json(incoming_json):
    """ OTW (over the wire) json objects can be ran through the model
        returns an array of 1's or 0's for the incoming json
    """
    model = ConcatPoolingGRUAdaptive(model_vocab_size, embedding_dim, rnn_hidden, n_out)
    predict_ds = SentimentDataset(max_vocab_size=vocab_size)
    predict_ds.load_vocabulary()
    predict_ds.load_json_as_df(incoming_json) #! call json.dumps(test_d) if testing
    model.import_dict()
    pred_dl = SentimentDataset.get_pdl(predict_ds)
    return ConcatPoolingGRUAdaptive.prediction(model, pred_dl)