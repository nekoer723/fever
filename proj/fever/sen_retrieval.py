import os
import argparse
# import torch
# from pytorch_pretrained_bert.tokenization import BertTokenizer
# from proj.fever.models import inference_model
# from proj.fever.data_loader import DataLoader, DataLoaderTest
# from proj.fever.bert_model import BertForSequenceEncoder
# import logging
# import json
# logger = logging.getLogger(__name__)



# def save_to_file(all_predict, outpath):
#     with open(outpath, "w") as out:
#         for key, values in all_predict.items():
#             sorted_values = sorted(values, key=lambda x:x[-1], reverse=True)
#             data = json.dumps({"id": key, "evidence": sorted_values[:5]})
#             out.write(data + "\n")



def eval_model(model, validset_reader):
    model.eval()
    all_predict = dict()
    for inp_tensor, msk_tensor, seg_tensor, ids, evi_list in validset_reader:
        probs = model(inp_tensor, msk_tensor, seg_tensor)
        probs = probs.tolist()
        assert len(probs) == len(evi_list)
        for i in range(len(probs)):
            print(evi_list[i])
            if ids[i] not in all_predict:
                all_predict['senList'] = []
            #if probs[i][1] >= probs[i][0]:
            all_predict['senList'].append(evi_list[i] + [probs[i]])
    return all_predict

parser = argparse.ArgumentParser()
parser.add_argument('--test_path', help='train path',default='data/testwxj.json')
parser.add_argument('--name', help='train path',default='mytest.json')
parser.add_argument("--batch_size", default=32, type=int, help="Total batch size for training.")
parser.add_argument('--outdir', help='path to output directory',default='data/senoutput/')
parser.add_argument('--bert_pretrain', default='bert_base')
parser.add_argument('--checkpoint', default='checkpoint/model.best.pt')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout.')
parser.add_argument('--no-cuda', action='store_true', default=True, help='Disables CUDA training.')
parser.add_argument("--bert_hidden_dim", default=768, type=int, help="Total batch size for training.")
parser.add_argument("--layer", type=int, default=1, help='Graph Layer.')
parser.add_argument("--num_labels", type=int, default=3)
parser.add_argument("--evi_num", type=int, default=5, help='Evidence num.')
parser.add_argument("--threshold", type=float, default=0.0, help='Evidence num.')
parser.add_argument("--max_len", default=120, type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                            "longer than this will be truncated, and sequences shorter than this will be padded.")
args1 = parser.parse_args(args=[])



