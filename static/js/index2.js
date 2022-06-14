$("#btn-fever2").on("click", function () {
    var claim = $("input[name='claim2']").val()
    // 验证claim是否为空
    if (!claim) {
        alert("请输入声明")
        return
    }
    $.ajax({
        type: 'POST',
        url: '/fever2',
        data: {text: claim},
        dataType: 'json',
        beforeSend: function () {
            // 清空doc-res
            $('#btn-fever2').html('正在进行事实验证，大约需要2分钟...');
            $('#btn-fever2').attr('disabled', true);
        }, error: function () {
            $('#btn-fever2').html('请求失败,请重试');
            $('#btn-fever2').attr('disabled', false);
        }, success: function (res) {
            $(document.getElementById("btn-fever2")).html('事实验证');
            $(document.getElementById("btn-fever2")).attr('disabled', false);
            if(res.result[0].label == 'SUPPORTS'){
                $(document.getElementById("btn-fever2")).attr("class","btn  ms-2 btn-sm btn-success");
                $(document.getElementById("btn-fever2")).text(res.result[0].label);
            }else {
                $(document.getElementById("btn-fever2")).attr("class","btn  ms-2 btn-sm btn-danger");
                $(document.getElementById("btn-fever2")).text(res.result[0].label);
            }
        }
    })
})