//import { dragdrop } from './imagehelper.js'
//idea from https://github.com/shinokada/fastapi-drag-and-drop

var tab_de = '<table style="border-collapse:collapse;border-spacing:0"><tr><th style="font-family:serif !important;;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;background-color:#f8ff00;color:#333333" rowspan="2">Grad</th><th style="font-family:serif !important;;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;background-color:#f8ff00;color:#333333" colspan="4">Modul</th></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;background-color:#f8ff00;color:#333333">Symmetrie</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;background-color:#f8ff00;color:#333333">Auge</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;background-color:#f8ff00;color:#333333">Mund</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;background-color:#f8ff00;color:#333333">Stirn</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;background-color:#ffffc7">I</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">normal</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">complete</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">normal</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">normal</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;background-color:#ffffc7">II</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">normal</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">complete</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">min_asymm</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">normal</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;background-color:#ffffc7">III</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">normal</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">complete</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">min_asymm</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">min_asymm</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;background-color:#ffffc7">IV</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">normal</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">incomplete</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">asymm</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">none</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;background-color:#ffffc7">V</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">asymm</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">incomplete</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">asymm</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">none</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;background-color:#ffffc7">VI</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">none</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">incomplete</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">none</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">none</td></tr></table>'


var tab_en = '<table style="border-collapse:collapse;border-spacing:0"><tr><th style="font-family:serif !important;;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;background-color:#f8ff00;color:#333333" rowspan="2">Grade</th><th style="font-family:serif !important;;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;background-color:#f8ff00;color:#333333" colspan="4">Module</th></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;background-color:#f8ff00;color:#333333">Symmetry</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;background-color:#f8ff00;color:#333333">Eye</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;background-color:#f8ff00;color:#333333">Mouth</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;background-color:#f8ff00;color:#333333">Forehead</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;background-color:#ffffc7">I</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">normal</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">complete</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">normal</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">normal</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;background-color:#ffffc7">II</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">normal</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">complete</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">min_asymm</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">normal</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;background-color:#ffffc7">III</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">normal</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">complete</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">min_asymm</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">min_asymm</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;background-color:#ffffc7">IV</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">normal</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">incomplete</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">asymm</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">none</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;background-color:#ffffc7">V</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">asymm</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">incomplete</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">asymm</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">none</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;background-color:#ffffc7">VI</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">none</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">incomplete</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">none</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal">none</td></tr></table>'

var set_lang = "de"
var languages = {
  "de": {
    "upload_button": "Hochladen",
    "main_drag_drop": 'Dateien hier Reinziehen <br> oder <br> Hochladen anklicken',
    "drag_here": "Hier Reinziehen",
    "drop": "Ablegen",
    "processing": "Bearbeite Patienten ...",
    "agb_label": "<b>Um Weiter fortzufahren müssen sie die Bedingungen Akzeptieren.</b>",
    "explanation_header": "Ermittlung des Grades der Fazialisparese nach House-Brackmann",
    "explanation_text": "Testumgebung der API Verbundung. <br> \
                         Dabei werden die hineingezogenen Images auf den Server <b>unverschlüsselt</b> transveriert. <br> \
                         Nach der Berechnung werden selbsständig alle Bilder vom Server gelöscht. <br> \
                         Dieses System soll eine unabhängige Prognose liefern. \
                         Es handelt sich um eine allgemeine Einschätzung und keine medizinische Klassifizierung! <br>\
                         <br> \
                         Angepasste House-Brackmann Tabelle:                                    <br> \
                         " + tab_de + " \
                         <br> \
                         Pro Patient werden 9 Bilder benötigt, die dieser Codierung und der Ordnerstruktur entsprechen: <br> \
                         ├── 001        &emsp;  #Ordner Patient 1                             <br> \
                         │   ├── 01.jpg &emsp;&emsp;&emsp; #Bild Ruhender Gesichtsausdruck    <br> \
                         │   ├── 02.jpg &emsp;&emsp;&emsp; #Bild Augenbraun heben             <br> \
                         │   ├── 03.jpg &emsp;&emsp;&emsp; #Bild Lächeln, geschlossener Mund  <br> \
                         │   ├── 04.jpg &emsp;&emsp;&emsp; #Bild geöffneter Mund              <br> \
                         │   ├── 05.jpg &emsp;&emsp;&emsp; #Bild Lippen schürzen, Duckface    <br> \
                         │   ├── 06.jpg &emsp;&emsp;&emsp; #Bild Augenschluss, leicht         <br> \
                         │   ├── 07.jpg &emsp;&emsp;&emsp; #Bild Augenschuss forciert         <br> \
                         │   ├── 08.jpg &emsp;&emsp;&emsp; #Bild Nase rümpfen                 <br> \
                         │   └── 09.jpg &emsp;&emsp;&emsp; #Bild Depression Unterlippe        <br> \
                         ├── 002        &emsp;  #Ordner Patient 2                             <br> \
                         │   ├── 01.jpg &emsp;&emsp;&emsp; #Bild Ruhender Gesichtsausdruck    <br> \
                         │   ├── 02.jpg &emsp;&emsp;&emsp; #Bild Augenbraun heben             <br> \
                         │   ├── 03.jpg &emsp;&emsp;&emsp; #Bild Lächeln, geschlossener Mund  <br> \
                         │   ├── 04.jpg &emsp;&emsp;&emsp; #Bild geöffneter Mund              <br> \
                         │   ├── 05.jpg &emsp;&emsp;&emsp; #Bild Lippen schürzen, Duckface    <br> \
                         │   ├── 06.jpg &emsp;&emsp;&emsp; #Bild Augenschluss, leicht         <br> \
                         │   ├── 07.jpg &emsp;&emsp;&emsp; #Bild Augenschuss forciert         <br> \
                         │   ├── 08.jpg &emsp;&emsp;&emsp; #Bild Nase rümpfen                 <br> \
                         │   └── 09.jpg &emsp;&emsp;&emsp; #Bild Depression Unterlippe        <br> \
                         └── ...         &emsp;  #Ordner Patient ...                          <br> <br> \
                         Unterstütze Formate für die Bilder sind ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo', 'heic'] <br> \
                         ",
    "upload_window": "Hochladen",
    "result_window": "Ergebnisse",
  },
  "en": {
    "upload_button": "Upload",
    "main_drag_drop": 'Drag and Drop file here <br> Or <br> Click to Upload',
    "drag_here": "Drag here",
    "drop": "Drop",
    "processing": "Processing Patients ...",
    "agb_label": "<b>I Accept the Conditions.</b>",
    "explanation_header": "Facial Palsy grade detection with House-Brackmann",
    "explanation_text": "Test environment of the API connection. <br> \
                         Uploaded images get transfered to the Server  <b>unencrypted</b> . <br> \
                         After the calculation, all images are automatically deleted from the Server. <br> \
                         This system is intended to provide an independent forecast. \
                         It is a general assessment and not a medical classification! <br> \
                         <br> \
                         Adjusted House-Brackmann Table:                                    <br> \
                         " + tab_en + " \
                         <br> \
                         9 images are required per patient, which correspond to this coding and the folder structure: <br> \
                         ├── 001        &emsp;  #Folder Patient 1                      <br> \
                         │   ├── 01.jpg &emsp;&emsp;&emsp; #Image Resting Face         <br> \
                         │   ├── 02.jpg &emsp;&emsp;&emsp; #Image Lift Eyebrown        <br> \
                         │   ├── 03.jpg &emsp;&emsp;&emsp; #Image Smile, closed Mouth  <br> \
                         │   ├── 04.jpg &emsp;&emsp;&emsp; #Image Smile, opend Mouth   <br> \
                         │   ├── 05.jpg &emsp;&emsp;&emsp; #Image Duckface             <br> \
                         │   ├── 06.jpg &emsp;&emsp;&emsp; #Image Close Eye, easy      <br> \
                         │   ├── 07.jpg &emsp;&emsp;&emsp; #Image Close Eye, forced    <br> \
                         │   ├── 08.jpg &emsp;&emsp;&emsp; #Image Wrinkle nose         <br> \
                         │   └── 09.jpg &emsp;&emsp;&emsp; #Image Lower Lip Depression <br> \
                         ├── 002       &emsp;  #Folder Patient 2                       <br> \
                         │   ├── 01.jpg &emsp;&emsp;&emsp; #Image Resting Face         <br> \
                         │   ├── 02.jpg &emsp;&emsp;&emsp; #Image Lift Eyebrown        <br> \
                         │   ├── 03.jpg &emsp;&emsp;&emsp; #Image Smile, closed Mouth  <br> \
                         │   ├── 04.jpg &emsp;&emsp;&emsp; #Image Smile, opend Mouth   <br> \
                         │   ├── 05.jpg &emsp;&emsp;&emsp; #Image Duckface             <br> \
                         │   ├── 06.jpg &emsp;&emsp;&emsp; #Image Close Eye, easy      <br> \
                         │   ├── 07.jpg &emsp;&emsp;&emsp; #Image Close Eye, forced    <br> \
                         │   ├── 08.jpg &emsp;&emsp;&emsp; #Image Wrinkle nose         <br> \
                         │   └── 09.jpg &emsp;&emsp;&emsp; #Image Lower Lip Depression <br> \
                         └── ...         &emsp;  #Folder Patient ...                   <br> <br> \
                         Allowed Formats for the Images are ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo', 'heic'] <br> \
                         ",
    "upload_window": "Upload",
    "result_window": "Results",
    "upload_files": "Uploading Filed..."
  }
}

$(function() {
  $('#drag_drop_zone').hide()
  $('#result_output').hide()

  $("#howto").html(languages[set_lang]["main_drag_drop"])
  $("#upload_button").text(languages[set_lang]["upload_button"])
  $("#agb_label").html(languages[set_lang]["agb_label"])
  $("#explanation_header").text(languages[set_lang]["explanation_header"])
  $("#explanation_text").html(languages[set_lang]["explanation_text"])
  $("#upload_window").text(languages[set_lang]["upload_window"])
  $("#result_window").text(languages[set_lang]["result_window"])



  //#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
  //Send Data to server
  function preparedata(file) {
    let data = {
      'winWidth': 1,
      'imgWidth': 2,
      'imgHeight': 3
    };

    let jdata = JSON.stringify(data);
    console.log('-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-')
    console.log("Preparing ...")
    console.log(file)

    var formData = new FormData();

    for (var x = 0; x < file.length; x++) {
      formData.append("files", file[x]);
    }

    $("#howto").text(languages[set_lang]["processing"]);
    $("#upload_button").hide();

    $.ajax({
      url: '/api/upload/',
      type: 'post',
      data: formData,
      contentType: false,
      processData: false,
      success: function(data) {
        console.log(data)
        updatetags(data);
      }
    });
  }

  function updatetags(data) {
    var obj = JSON.stringify(data, null, 3)
    console.log(obj)
    $("#output").html(obj);

    $("#upload_button").show();
    $("#howto").html(languages[set_lang]["main_drag_drop"])
  }



  //#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
  //Drag-And-Drop
  // preventing page from redirecting
  $("html").on("dragover", function(e) {
    e.preventDefault();
    e.stopPropagation();
    $("#howto").text(languages[set_lang]["drag_here"]);
  });

  $("html").on("drop", function(e) {
    e.preventDefault();
    e.stopPropagation();
  });

  // Drag over
  $('.upload-area').on('dragover', function(e) {
    e.stopPropagation();
    e.preventDefault();
    $("#howto").text(languages[set_lang]["drop"]);
  });



  // Drop
  $('.upload-area').on('drop', function(e) {
      e.stopPropagation();
      e.preventDefault();
      $("#howto").text(languages[set_lang]["upload_files"]);
      let file = e.originalEvent.dataTransfer.files;



      var items = event.dataTransfer.items;
      var up = [];





    getFilesFromWebkitDataTransferItems(items).then(files => {
      for (var i = 0; i < files.length; i++) {
        console.log(files[i])
        up[i] = new File([files[i]], files[i].filepath);
      }

      preparedata(up);
    })

    console.log("done drop.")
  });

//#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
//Button
// Open file selector on div click
$("#uploadfile").click(function() {
  $("#file").click();
});

// file selected
$("#file").change(function() {
  $("#howto").text(languages[set_lang]["upload_files"]);
  preparedata($('#file')[0].files);
});



//#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

function getFilesFromWebkitDataTransferItems(dataTransferItems) {
  function traverseFileTreePromise(item, path = '') {
    return new Promise(resolve => {
      if (item.isFile) {
        item.file(file => {
          file.filepath = path + file.name //save full path
          files.push(file)
          resolve(file)
        })
      } else if (item.isDirectory) {
        let dirReader = item.createReader()
        dirReader.readEntries(entries => {
          let entriesPromises = []
          for (let entr of entries)
            entriesPromises.push(traverseFileTreePromise(entr, path + item.name + "/"))
          resolve(Promise.all(entriesPromises))
        })
      }
    })
  }

  let files = []
  return new Promise((resolve, reject) => {
    let entriesPromises = []
    for (let it of dataTransferItems)
      entriesPromises.push(traverseFileTreePromise(it.webkitGetAsEntry()))
    Promise.all(entriesPromises)
      .then(entries => {
        //console.log(entries)
        resolve(files)
      })
  })
}





});








value = true

function checkit() {
  if (value == true) {
    $('#result_output').show()
    $('#drag_drop_zone').show()
    console.log(value)
  } else {
    $('#result_output').hide()
    $('#drag_drop_zone').hide()
    console.log(value)
  }
  value = !value
}

function lang_change(selectObject) {
  var value = selectObject.value;
  console.log(value);
  set_lang = value

  $("#howto").html(languages[set_lang]["main_drag_drop"])
  $("#upload_button").text(languages[set_lang]["upload_button"])
  $("#agb_label").html(languages[set_lang]["agb_label"])
  $("#explanation_header").text(languages[set_lang]["explanation_header"])
  $("#explanation_text").html(languages[set_lang]["explanation_text"])
  $("#upload_window").text(languages[set_lang]["upload_window"])
  $("#result_window").text(languages[set_lang]["result_window"])
}
