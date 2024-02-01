document.addEventListener("DOMContentLoaded", function(){
    var del_click = document.getElementById("del-btn");
    var gen_click = document.getElementById("gen-btn");
    var text_region = document.getElementById("text");
    var hiddenText = document.getElementById("hidden-text");
    var show_hint_btn = document.getElementById("show_hint");
    var hint = document.getElementById("hint");
    var hide_hint_btn = document.getElementById("hide_hint");
    
    del_click.addEventListener('click', function(){
        text_region.innerText = '';
        hiddenText.value = '';
    });

    text_region.addEventListener('input', function() {
        hiddenText.value = text_region.innerText;
    });

    show_hint_btn.addEventListener('click', function(){
        hint.style.display = 'flex';
        hide_hint_btn.style.display = 'block';
        show_hint_btn.style.display = 'none';
    });

    hide_hint_btn.addEventListener('click', function(){
        hint.style.display = 'none';
        hide_hint_btn.style.display = 'none';
        show_hint_btn.style.display = 'block';
    });
});
