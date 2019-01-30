var el = x => document.getElementById(x);

function showPicker(inputId) { el('file-input').click(); }

function showPicked(input) {
    el('upload-label').innerHTML = input.files[0].name;
    var reader = new FileReader();
    reader.onload = function (e) {
        el('image-picked').src = e.target.result;
        el('image-picked').className = '';
    }
    reader.readAsDataURL(input.files[0]);
}

function analyze() {
    var uploadFiles = el('file-input').files;
    if (uploadFiles.length != 1) alert('Please select 1 file to analyze!');

    el('analyze-button').innerHTML = 'Analyzing...';
    var xhr = new XMLHttpRequest();
    var loc = window.location
    xhr.open('POST', `${loc.protocol}//${loc.hostname}:${loc.port}/analyze`, true);
    xhr.onerror = function() {alert (xhr.responseText);}
    xhr.onload = function(e) {
        if (this.readyState === 4) {
            var response = JSON.parse(e.target.responseText);
            // el('result-label').innerHTML = `Result = ${response['result']}`;
            if (response['class'] < 2) {
            if (response['class'] == 0) {
                bgc_class = 'palmer'
            }
            else {
                bgc_class = 'tommy-atkins'
            }
            el('result-label').innerHTML = `<div class="recognized"><span class="${bgc_class}">${response['prediction']}</span>
                                            <br />confidence of ${response['prob']}</div>
                                            <div class="probs">[ ${response['probs'][0]}, ${response['probs'][1]} ]</div>`;
            } else {
            el('result-label').innerHTML = `<div class="not-recognized">Image unrecognized<br /></div>
                                            <div class="probs">[ ${response['probs'][0]}, ${response['probs'][1]} ]</div>`;
            }
        }
        el('analyze-button').innerHTML = 'Analyze';
    }

    var fileData = new FormData();
    fileData.append('file', uploadFiles[0]);
    xhr.send(fileData);
}

