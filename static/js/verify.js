function verify(name){
    var evidence = $(document.getElementById("textarea"+name)).val();
    var claim = $("input[name='claim']").val()
    $.ajax({
        url:"/verify",
        method:"POST",
        data:{
            "claim":claim,
            "evidence":evidence,
        },
        beforeSend:function (){
            $(document.getElementById("ver"+name)).html('正在验证...');
            $(document.getElementById("ver"+name)).attr("class","btn  ms-2 btn-sm btn-warning");
            $(document.getElementById("ver"+name)).attr('disabled', true);
        },error:function (){
            $(document.getElementById("ver"+name)).html('验证失败请重试');
            $(document.getElementById("ver"+name)).attr("class","btn  ms-2 btn-sm btn-danger");
            $(document.getElementById("ver"+name)).attr('disabled', false);
        },success:function (res){
            console.log(res);
            $(document.getElementById("ver"+name)).html('验证');
            $(document.getElementById("ver"+name)).attr('disabled', false);
            if(res.result[0].label == 'SUPPORTS'){
                $(document.getElementById("ver"+name)).attr("class","btn  ms-2 btn-sm btn-success");
                $(document.getElementById("ver"+name)).text(res.result[0].label);
            }else {
                $(document.getElementById("ver"+name)).attr("class","btn  ms-2 btn-sm btn-danger");
                $(document.getElementById("ver"+name)).text(res.result[0].label);
            }
        }
    })
}