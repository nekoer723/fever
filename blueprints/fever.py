from flask import Blueprint,render_template,request,jsonify
from blueprints.forms import ClaimForm
from proj.tasks import cverify,cdocRetrieval,cdocText,csenRetrieval
import time

bp = Blueprint("fever", __name__, url_prefix="/")


@bp.route('/')
def index():
    return render_template('index.html')


# @bp.route('/retrieval',methods=['POST'])
# def retrieval():
#     form = ClaimForm(request.form)
#     claim = form.claim.data
#     print(claim)
#     if claim == "jack is a cat":
#         evidences = retrievalsentens()
#         # return jsonify({"claim": claim, "evidences":evidences})
#         print(evidences)
#         return render_template('index.html', evidences=evidences, claim=claim)
#     else:
#         evidences = []
#         return render_template('index.html', evidences=evidences, claim="")


@bp.route('/verify',methods=['POST'])
def verify():
    form = ClaimForm(request.form)
    claim = form.claim.data
    evidence = form.evidence.data
    ver = cverify.delay(claim,evidence)
    for i in range(10):
        if ver.ready():
            result = ver.get()
            return jsonify({"claim": claim, "evidences":evidence, "result":result})
        else:
            time.sleep(1)
    return jsonify({"claim": claim, "evidences":evidence, "result":"waiting"})


@bp.route('/doc_retrieval',methods=['POST'])
def docRetrieval():
    claim = request.form['text']
    docR = cdocRetrieval.delay(claim,5)
    for i in range(10):
        if docR.ready():
            document = docR.get()
            return jsonify(document)
        else:
            time.sleep(1)
    return jsonify({"code": "waiting"})


@bp.route('/doc_text',methods=['POST'])
def docText():
    doc_name = request.form['doc_id']
    print(doc_name)
    docT = cdocText.delay(doc_name)
    for i in range(10):
        if docT.ready():
            document = docT.get()
            return jsonify(document)
        else:
            time.sleep(1)
    return jsonify({"doc_text":None})

@bp.route('/sen_retrieval',methods=['POST'])
def senRetrieval():
    claim = request.form['claim']
    context = request.form['sentences']
    # 去掉context中的空格
    # 将context分割成sentences
    sentences = context.split(".")
    # 去掉句子中的空字符串和长度为1的字符串
    sentences = [sentence for sentence in sentences if sentence != "" and len(sentence) > 5]
    # sentences = [sentence for sentence in sentences if sentence]
    print('句子数量')
    print(len(sentences))
    docR = csenRetrieval.delay(claim,sentences)
    for i in range(1000):
        if docR.ready():
            senList = docR.get()
            return jsonify(senList)
        else:
            time.sleep(1)
    return jsonify({"code": "waiting"})

@bp.route('/index2')
def index2():
    return render_template('index2.html')

@bp.route('/fever2', methods=['POST'])
def fever2():
    claim = request.form['text']
    docR = cdocRetrieval.delay(claim,5)
    for i in range(10):
        if docR.ready():
            document = docR.get()
            break
        else:
            time.sleep(1)
    document = document['doc_contexts'][0]
    # 将document分割成长度大于3的sentences
    sentences = document.split(".")
    sentences = [sentence for sentence in sentences if sentence != "" and len(sentence) > 5]
    docR = csenRetrieval.delay(claim,sentences)
    for i in range(1000):
        if docR.ready():
            senList = docR.get()
            break
        else:
            time.sleep(1)
    evidence = senList[0][0]
    ver = cverify.delay(claim,evidence)
    for i in range(10):
        if ver.ready():
            result = ver.get()
            return jsonify({"claim": claim, "evidences":evidence, "result":result})
        else:
            time.sleep(1)
    return jsonify({"claim": claim, "evidences":evidence, "result":'fail'})

