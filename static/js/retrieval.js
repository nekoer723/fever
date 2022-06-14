function getSentence(name,i) {
    var myselect = document.getElementById("select" + i);
    var index = myselect.selectedIndex;
    $(document.getElementById("textarea"+name)).val(myselect.options[index].text);
}
// function retrieval(){
//     // alert("ssss")
//     $("#claim-btn").on("click",function () {
//         var claim = $("input[name='claim']").val()
//         if(!claim){
//             alert("请输入声明")
//             return
//         }
//     })
// }
var contextg = [];
function doc_text(doc_id) {
    name = unescape(doc_id)
    $.ajax({
        type: 'POST',
        url: '/doc_text',
        data: { doc_id: name },
        dataType: 'json',
        beforeSend: function () {
            $(document.getElementById('doc'+doc_id)).html('正在获取文档...');
            $(document.getElementById('doc'+doc_id)).attr('disabled', true);
        }, error: function () {
            $(document.getElementById('doc'+doc_id)).html('请求失败,请重试');
            $(document.getElementById('doc'+doc_id)).attr('disabled', false);
        }, success: function (res) {
            $(document.getElementById('doc'+doc_id)).html('title：' + name);
            $(document.getElementById('doc'+doc_id)).attr('disabled', false);
            $(document.getElementById('doc-text'+doc_id)).html(res['doc_text'])
        }
    })
}
$("#btn-search").on("click", function () {
    var claim = $("input[name='claim']").val()
    // 验证claim是否为空
    if (!claim) {
        alert("请输入声明")
        return
    }
    $.ajax({
        type: 'POST',
        url: '/doc_retrieval',
        data: {text: claim},
        dataType: 'json',
        beforeSend: function () {
            // 清空doc-res
            $('#doc-res').html('');
            $('#btn-search').html('正在检索文档...');
            $('#btn-search').attr('disabled', true);
        }, error: function () {
            $('#btn-search').html('请求失败,请重试');
            $('#btn-search').attr('disabled', false);
        }, success: function (res) {
           //res=res['answer']
            console.log(res)
            $('#btn-search').html('文档检索');
            $('#btn-search').attr('disabled', false);
            var names = res['doc_names']
            var scores = res['doc_scores']
            var contexts = res['doc_contexts']
            console.log(res)
            var escape_claim = escape(claim)
            for (var i = 0; i < names.length; i++) {
                var name = names[i]
                // 去掉name里的空格和
                var escape_name = escape(name)
                var nameCollapse = escape_name.replace(/\s/g, '').replace(/%/g, '')
                var score = scores[i]
                var context = contexts[i]
                contextg[i] = context
                context = context.replace(/\n/g, '<br>')
                var sen_body ='<div class="container my-4">' +
                    '                    <div class="row">' +
                    '                        <div class="col-md-9">' +
                    '                            <select id="select'+i+'" class="form-select d-inline " aria-label="Default select example" onchange="getSentence(\''+escape_name+'\',\''+i+'\')">\n' +
                    '                                <option selected >选择检索出的证据</option>' +
                    '                                <option value="1">'+name+'</option>' +
                    '                                <option value="2">Two</option>' +
                    '                                <option value="3">Three</option>' +
                    '                            </select>' +
                    '                        </div>' +
                    '                        <div class="col-md-3">' +
                    '                            <button id="sen'+i+'"class="btn btn-primary" onclick="sen_retrieval(\''+escape_claim+'\',\''+i+'\')"> 句子检索</button>' +
                    '                        </div>' +
                    '                    </div>' +
                    '                </div>' +
                    '                <div class="container my-4">' +
                    '                    <div class="row">' +
                    '                        <div class="col-md-9">' +
                    '                            <div class="card">' +
                    '                                <div class="card-body">' +
                    '                                    <textarea class="form-control" name="" id="textarea'+escape_name+'" rows="6"></textarea>' +
                    '                                </div>' +
                    '                            </div>' +
                    '                        </div>' +
                    '                        <div class="col-md-3">' +
                    '                            <button id="ver'+escape_name+'" class="btn btn-primary" onclick="verify(\''+escape_name+'\')">事实验证</button>' +
                    '                        </div>' +
                    '                    </div>' +
                    '                </div>'
                var doc_btn = '<div class="accordion-item"> <h2 class="accordion-header" id="heading'
                    +escape_name+'"> <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse"data-bs-target="#collapse'
                    +nameCollapse+'" aria-expanded="false" aria-controls="collapse'
                    +nameCollapse+'">'
                    +name+
                    "  得分:"
                    +score+'</button> </h2> <div id="collapse'+
                    nameCollapse+'" class="accordion-collapse collapse" aria-labelledby="heading'
                    +escape_name+'"data-bs-parent="#doc-res"> <div class="accordion-body">'+context+'</div>'+sen_body+'</div></div>'
                // var doc_text = '<div class="card card-body d-block pb-2 invisible" id="doc-text' + escape(doc_id) + '"></div>'

                $("#doc-res").append(doc_btn)
                // $("#doc-res").append(doc_text)
            }
        }
    })
})

function sen_retrieval(claim,i){
    // 去掉context里的换行符
    var context = contextg[i].replace(/\n/g, '')
    // 将context分割成句子
    var sentences = context.split('.')
    claim = unescape(claim)
    console.log(sentences);
    $.ajax({
        type: 'POST',
        url: '/sen_retrieval',
        data: {sentences: context, claim: claim},
        dataType: 'json',
        beforeSend: function () {
            // 改变当前按钮的文字
            $('#sen'+i).html('正在检索句子...');
            $('#sen'+i).attr('disabled', true);
        }, error: function () {
            $('#sen'+i).html('请求失败,请重试');
            $('#sen'+i).attr('disabled', false);
        }, success: function (res) {
            console.log(res);
            $('#sen'+i).html('句子检索');
            $('#sen'+i).attr('disabled', false);
            var sentences=  '                            <option selected >选择检索出的证据</option>' +
                            '                                <option value="1">'+res[0][0]+res[0][1]+'</option>' +
                            '                                <option value="2">'+res[1][0]+res[1][1]+'</option>' +
                            '                                <option value="3">'+res[2][0]+res[2][1]+'</option>' 
           $('#select'+i).html(sentences)
        }
    })
}
