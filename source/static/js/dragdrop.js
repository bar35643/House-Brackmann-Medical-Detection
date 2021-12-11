//import { dragdrop } from './imagehelper.js'
//idea from https://github.com/shinokada/fastapi-drag-and-drop

var set_lang = "de"
var languages = {
  "de": {
    "upload_button": "Hochladen",
    "main_drag_drop": 'Dateien hier Reinziehen <br> oder <br> Hochladen anklicken',
    "drag_here": "Hier Reinziehen",
    "drop": "Ablegen",
    "agb_label": "Ich akzeptiere die Allgemeinen Geschäftbedingungen, die Datenschutzerklärung und die Widerrufsbelehrung",
    "explanation_header": "HOWTO",
    "explanation_text": "TODO write exp text",
    "upload_window": "Hochladen",
    "result_window": "Ergebnisse",
  },
  "en": {
    "upload_button": "Upload",
    "main_drag_drop": 'Drag and Drop file here <br> Or <br> Click to Upload',
    "drag_here": "Drag here",
    "drop": "Drop",
    "agb_label": "Ich akzeptiere die Allgemeinen Geschäftbedingungen, die Datenschutzerklärung und die Widerrufsbelehrung",
    "explanation_header": "HOWTO",
    "explanation_text": "TODO write exp text",
    "upload_window": "Upload",
    "result_window": "Results",
    "upload_files": "Uploading Filed..."
  }
}

$(function() {
  $('#drag_drop_zone').hide()
  $("#howto").html(languages[set_lang]["main_drag_drop"])
  $("#upload_button").text(languages[set_lang]["upload_button"])
  $("#agb_label").text(languages[set_lang]["agb_label"])
  $("#explanation_header").text(languages[set_lang]["explanation_header"])
  $("#explanation_text").text(languages[set_lang]["explanation_text"])
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
    getFilesFromWebkitDataTransferItems(items).then(files => {
      for (var i = 0; i < files.length; i++) {
        files[i] = new File([new Blob()], files[i].filepath);
      }

      preparedata(files);
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



function updatetags(data) {
  let show = document.createElement("h3");
  $("#output").text(data.result);

  $("#howto").html(languages[set_lang]["main_drag_drop"])
}






value = true

function checkit() {
  if (value == true) {
    $('#drag_drop_zone').show()
    console.log(value)
  } else {
    $('#drag_drop_zone').hide()
    console.log(value)
  }
  value = !value
}
