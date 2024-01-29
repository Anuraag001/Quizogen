document.addEventListener("DOMContentLoaded", function(){
    var del_click = document.getElementById("del-btn");
    var gen_click = document.getElementById("gen-btn");
    var text_region = document.getElementById("text");
    var hiddenText = document.getElementById("hidden-text");

    del_click.addEventListener('click', function(){
        text_region.innerText = '';
        hiddenText.value = '';
        console.log("clicked");
    });

    text_region.addEventListener('input', function() {
        hiddenText.value = text_region.innerText;
    });
});
