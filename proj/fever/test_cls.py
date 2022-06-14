import argparse
import logging
import os
import sys

from torch import nn

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_PATH)
sys.path.insert(1, BASE_PATH + '/../')

import torch
from transformers import BertForSequenceClassification
torch.device('cpu')

class QGgenerate(nn.Module):
    def __init__(self, args):
        super(QGgenerate, self).__init__()
        self.args = args
        self.model = BertForSequenceClassification.from_pretrained(args.model)
        self.linear = torch.nn.Linear(768, 2)

    def forward(self, inputs, return_logits=None):
        outputs = self.model(input_ids=inputs["input_ids"])
        logits = outputs.logits
        loss = outputs.loss
        if return_logits:
            return logits
        else:
            return loss


def evaluate(model, dataloader):
    model.eval()
    result = []
    with torch.no_grad():
        for step, input in enumerate(dataloader):
            outputs = model(input, return_logits=True)
            outputs = outputs.tolist()
            for i in range(len(outputs)):
                if outputs[i][0] > outputs[i][1]:
                    result.append({"label": 'SUPPORTS'})
                else:
                    result.append({"label": 'REFUTES'})
    return result


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='bert-model')
parser.add_argument("--seed", type=int, default=144)
parser.add_argument("--max_len", default=256, type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                         "longer than this will be truncated, and sequences shorter than this will be padded.")
parser.add_argument("--test_batch_size", default=8, type=int, help="Total batch size for predictions.")
parser.add_argument('--test_path', help='test path', default='data/test.json')
parser.add_argument('--test_result_path', help='test result path', default='data/res.json')
parser.add_argument("--soft_token_num", type=int, default=5)
parser.add_argument('--checkpoint', default='checkpoint/fever_finetune.ckpt', help='checkpoint')
parser.add_argument('--result_path', help='result path')
parser.add_argument("--template", type=str, default='LM')
parser.add_argument("--project_root", default="./save_model",
                    help="The project root in the file system, i.e. the absolute path of FactPrompt")
args = parser.parse_args(args=[])
if not os.path.exists(args.project_root):
    os.mkdir(args.project_root)
# 训练的日志
# handlers = [logging.FileHandler(os.path.abspath(args.project_root) + '/test_cls_log.txt'),
#             logging.StreamHandler()]
# logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
#                     datefmt='%d-%m-%Y %H:%M:%S', handlers=handlers)
model = QGgenerate(args)
if args.checkpoint is not None:
    model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device('cpu')), strict=False)
model = model
model.eval()
