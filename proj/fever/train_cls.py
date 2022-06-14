import json
import logging
import sys
import os
from typing import Optional

import jsonlines
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from tqdm import tqdm
import torch
import argparse
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_PATH)
sys.path.insert(1, BASE_PATH + '/../')
import time
logger = logging.getLogger(__name__)
import random
from transformers import BertConfig, BertTokenizer, BertForMaskedLM, AdamW, get_linear_schedule_with_warmup, Adafactor, \
    get_constant_schedule_with_warmup, BertModel, AutoModel, BertForSequenceClassification

from dataloader_cls import GenDataLoader

class QGgenerate(nn.Module):
    def __init__(self, args):
        super(QGgenerate, self).__init__()
        self.args = args
        # self.model = AutoModel.from_pretrained(args.model)
        self.model = BertForSequenceClassification.from_pretrained(args.model)
        #self.tokenizer = BertTokenizer.from_pretrained(args.model)
        self.linear = torch.nn.Linear(768, 2).cuda()
        # self.linear = torch.nn.Linear(1024, 2)
    def forward(self,inputs,return_logits=None):
        outputs = self.model(input_ids=inputs["input_ids"],labels=inputs["label"])
        logits = outputs.logits
        loss = outputs.loss
        if return_logits:
            return logits
        else:
            return loss
        # outputs = outputs.pooler_output
        # outputs = self.linear(outputs)



def set_seed(seed: Optional[int] = None):
    """set seed for reproducibility
     Args:
    seed (:obj:`int`): the seed to seed everything for reproducibility. if None, do nothing.
    """
    if seed:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        logger.info(f"Global seed set to {seed}")

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
    with jsonlines.open(args.valid_result_path, "w") as writer:
        for line in result:
            writer.write(line)


def load_full_context_with_ppl(data_path, ppl_result_path):
    with jsonlines.open(data_path) as reader:
        og_objs = [obj for obj in reader]
    with jsonlines.open(ppl_result_path) as reader:
        ppl_results = [obj for obj in reader]
    all_objs = []
    for obj, ppl in zip(og_objs, ppl_results):
        label = multi2binary[obj['label']]
        prelabel = multi2binary[ppl['label']]
        new_objs = {'prelabel': prelabel, 'label': label}
        all_objs.append(new_objs)
    return all_objs


def get_metric(objs):
    preds = [obj['prelabel'] for obj in objs]
    golds = [obj['label'] for obj in objs]
    acc = accuracy_score(golds, preds)
    f1_macro = f1_score(golds, preds, pos_label='false', average='macro')
    return {'acc': acc, 'f1_macro': f1_macro}


def train(model, args, trainset_reader, validset_reader):
    # loss_func = torch.nn.CrossEntropyLoss()
    loss_func = torch.nn.BCEWithLogitsLoss()
    tot_step = int(trainset_reader.dataset.total_num / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
    no_decay = ['bias',
                'LayerNorm.weight']  # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay))],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=args.prompt_lr)
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1,
        num_warmup_steps=args.warmup_steps, num_training_steps=tot_step)
    tot_loss = 0
    best_f1 = 0
    glb_step = 0
    actual_step = 0
    patience_counter = 0
    for epoch in range(int(args.num_train_epochs)):
        print(f"Begin epoch {epoch}")
        for step, inputs in enumerate(trainset_reader):
            model.train()
            loss = model(inputs, return_logits=False)
            tot_loss += loss.item()
            # targets = inputs['label']
            # loss = loss_func(outputs, targets.float())
            if args.gradient_accumulation_steps > 1:
                loss /= args.gradient_accumulation_steps
            loss.backward()

            actual_step += 1
            if actual_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                glb_step += 1
                optimizer1.step()
                scheduler1.step()
                optimizer1.zero_grad()
                logger.info('Epoch: {0}, Step: {1}, Loss: {2}'.format(epoch, actual_step, (tot_loss / actual_step)))
                if actual_step % args.gradient_accumulation_steps == 0 and glb_step > 0 and glb_step % args.eval_every_steps == 0:
                    logger.info('Start eval!')
                    evaluate(model, validset_reader)
                    testdata = load_full_context_with_ppl(args.valid_path, args.valid_result_path)
                    result = get_metric(testdata)
                    val_acc = result["acc"]
                    f1 = result["f1_macro"]
                    logger.info('Dev total acc: {0},F1:{1}'.format(val_acc, f1))
                    if f1 > best_f1:
                        torch.save(model.state_dict(), f"{args.project_root}/{args.result_file}.ckpt")
                        logging.info("Saved best epoch {0}, best acc {1}, F1 {2}".format(epoch, val_acc, f1))
                        best_f1 = f1
                #         patience_counter = 0
                #     else:
                #         patience_counter += 1
                # if patience_counter >= args.patience:
                #     logging.info("Early stopping...")
                #     return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--seed", type=int, default=144)
    parser.add_argument("--model", type=str, default='bert-base-uncased',
                        help="We test both t5 and t5-lm in this scripts, the corresponding tokenizerwrapper will be automatically loaded.")
    parser.add_argument("--project_root", default="./save_model",
                        help="The project root in the file system, i.e. the absolute path of OpenPrompt")
    parser.add_argument("--result_file", type=str)
    parser.add_argument("--prompt_lr", type=float, default=3e-5)
    parser.add_argument("--eval_every_steps", type=int, default=1000)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument('--train_path', help='/Users/wuyong/PycharmProjects/eva/data/bert_train.json')
    parser.add_argument('--valid_path', help='test path')
    parser.add_argument('--valid_result_path', help='test path')
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--train_batch_size", default=16, type=int, help="train_batch_size.")
    parser.add_argument("--valid_batch_size", default=16, type=int, help="valid_batch_size.")
    parser.add_argument("--max_len", default=256, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--warmup_steps", default=500, type=int, help="Linear warmup over warmup_steps.")
    args = parser.parse_args()
    if not os.path.exists(args.project_root):
        os.mkdir(args.project_root)
    handlers = [logging.FileHandler(os.path.abspath(args.project_root) + "/" + args.result_file + 'train_log.txt'),
                logging.StreamHandler()]
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                        datefmt='%d-%m-%Y %H:%M:%S', handlers=handlers)
    logger.info(args)
    set_seed(args.seed)
    model = QGgenerate(args)
    model = model.cuda()
    tokenizer = BertTokenizer.from_pretrained(args.model)
    trainset = GenDataLoader(args.train_path, args)
    validset = GenDataLoader(args.valid_path, args)
    train_sampler = RandomSampler(trainset)
    valid_sampler = SequentialSampler(validset)
    trainset_reader = DataLoader(
        dataset=trainset,
        num_workers=0,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=trainset.collate_fn,
        shuffle=False,
    )
    validset_reader = DataLoader(
        dataset=validset,
        sampler=valid_sampler,
        num_workers=0,
        batch_size=args.valid_batch_size,
        collate_fn=validset.collate_fn,
        shuffle=False,
    )
    multi2binary = {
        "true": "true",
        "false": "false",
        "REFUTES": "false",
        "SUPPORTS": "true",
        "REFUTED": "false",
        "SUPPORTED": "true",
        "CONTRADICT": "false",
        "SUPPORT": "true",
        "NOT ENOUGH INFO": "NOT ENOUGH INFO",
    }

    train(model, args, trainset_reader, validset_reader)

