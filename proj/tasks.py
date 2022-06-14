from proj.celery import celery
import time
from proj.fever.doc_retrieval import process,getText

@celery.task
def add(x, y):
    time.sleep(10)
    return x + y

@celery.task
def cdocText(doc_id):
    document = getText(doc_id)
    return document

@celery.task
def cdocRetrieval(claim, k):
    return process(claim, k)

@celery.task
def cverify(claim, evidence):
    from torch.utils.data import DataLoader
    from proj.fever.dataloader_cls import GenDataLoader
    from proj.fever.test_cls import args,model
    import torch
    testset = GenDataLoader(args, claim, evidence)
    test_reader = DataLoader(
        dataset=testset,
        num_workers=0,
        batch_size=args.test_batch_size,
        collate_fn=testset.collate_fn,
        shuffle=False,
    )
    model.eval()
    result = []
    with torch.no_grad():
        for step, input in enumerate(test_reader):
            outputs = model(input, return_logits=True)
            outputs = outputs.tolist()
            for i in range(len(outputs)):
                if outputs[i][0] > outputs[i][1]:
                    result.append({"label": 'SUPPORTS'})
                else:
                    result.append({"label": 'REFUTES'})
    return result

@celery.task
def csenRetrieval(claim, evidences):
    import os
    from proj.fever.sen_retrieval import args1
    from pytorch_pretrained_bert.tokenization import BertTokenizer
    from proj.fever.data_loader import DataLoaderTest
    from proj.fever.models import inference_model
    from proj.fever.bert_model import BertForSequenceEncoder
    import torch
    tokenizer = BertTokenizer.from_pretrained(args1.bert_pretrain, do_lower_case=False)
    validset_reader = DataLoaderTest(args1.test_path, tokenizer, args1, cuda=False, batch_size=args1.batch_size,sentences=evidences,claim=claim)
    def eval_model(model, validset_reader):
        model.eval()
        all_predict = dict()
        all_predict['senList']=[]
        for inp_tensor, msk_tensor, seg_tensor, ids, evi_list in validset_reader:
            # print(len(evi_list))
            probs = model(inp_tensor, msk_tensor, seg_tensor)
            probs = probs.tolist()
            # print(probs)
            # print(len(probs))
            assert len(probs) == len(evi_list)
            for i in range(len(probs)):
                all_predict['senList'].append([evi_list[i]] + [probs[i]])
        return all_predict
    if not os.path.exists(args1.outdir):
        os.mkdir(args1.outdir)
    args1.cuda = not args1.no_cuda and torch.cuda.is_available()
    bert_model = BertForSequenceEncoder.from_pretrained(args1.bert_pretrain)
    # bert_model = bert_model.cuda()
    model = inference_model(bert_model, args1)
    model.load_state_dict(torch.load(args1.checkpoint, map_location=torch.device('cpu'))['model'])
    predict_dict = eval_model(model, validset_reader)
    predict_dict['senList'].sort(key=lambda x:x[-1], reverse=True)
    # 返回前三个句子
    return predict_dict['senList'][:3]
    

