import torch
from torch.utils.data import Dataset
import json
from transformers import BertTokenizer

# logger = logging.getLogger()


class GenDataLoader(Dataset):
    def __init__(
            self,
            args,
            claim,
            evidence,
    ):
        """
        :param intput_dir: examples.jsonl ("pos_docid"/"neg_docid"); docid2doc.jsonl
        :param tokenizer: T5Tokenizer or None
        """
        self.tokenizer = BertTokenizer.from_pretrained(args.model)
        self.args = args
        self.claim = claim
        self.evidence = evidence
        self.max_len = args.max_len
        self.data = self.data_read()
        self.pad_tok = self.tokenizer.pad_token
        self.total_num = len(self.data)

    def data_read(self):
        data = []
        data.append([self.evidence, self.claim])
        return data

    def read_file(self, path):
        data = []
        with open(path) as fin:
            for line in fin:
                example = json.loads(line.strip())
                claim = example["claim"]
                evs_line = example["evidence"]
                # label = example["label"]
                # if label == 'SUPPORT' or label == 'SUPPORTS' or label == 'SUPPORTED':
                #    label = 0
                # else:
                #    label = 1
                # data.append([evs_line, claim, label])
                data.append([evs_line, claim])
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # example = self.data[index]
        # claim = example[1]
        # evs_line = example[0]
        # label = example[2]
        # print(evs_line)
        otherinput_tokens = self.tokenizer.tokenize(self.claim + '[SEP]')
        others_length = len(otherinput_tokens)
        evi_length = self.max_len - others_length
        eviinput_tokens = self.tokenizer.tokenize('[CLS]' + self.evidence)
        eviinput_tokens = eviinput_tokens[:evi_length]
        input_tokens = eviinput_tokens + otherinput_tokens
        inputs = input_tokens + [self.pad_tok] * (self.max_len - len(input_tokens))
        input_ids = self.tokenizer.convert_tokens_to_ids(inputs)
        # return {"input_ids": input_ids, "label": label}
        return {"input_ids": input_ids}

    def collate_fn(self, batch_data):
        input_ids = [data["input_ids"] for data in batch_data]
        # label = [data["label"] for data in batch_data]
        input_ids = torch.LongTensor(input_ids)
        # label = torch.LongTensor(label).cuda()
        # return {"input_ids": input_ids, "label": label}

        return {"input_ids": input_ids}
