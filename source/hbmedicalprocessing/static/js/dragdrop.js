//import { dragdrop } from './imagehelper.js'
//idea from https://github.com/shinokada/fastapi-drag-and-drop

var set_lang = "de"
var languages = {
  "de": {
    "upload_button": "Hochladen",
    "main_drag_drop": 'Dateien hier Reinziehen <br> oder <br> Hochladen anklicken',
    "drag_here": "Hier Reinziehen",
    "drop": "Ablegen",
    "agb_label": "<b>Um Weiter fortzufahren müssen sie die Bedingungen Akzeptieren.</b>",
    "explanation_header": "Ermittlung des Grades der Fazialisparese nach House-Brackmann",
    "explanation_text": "Testumgebung der API Verbundung. <br> \
                         Dabei werden die hineingezogenen Images auf den Server transveriert. <br> \
                         Nach der Berechnung werden selbsständig alle Bilder vom Server gelöscht. <br> \
                         <br> \
                         Dieses System soll eine unabhängige Prognose liefern. \
                         Es handelt sich um eine allgemeine Einschätzung und keine medizinische Klassifizierung! <br> \
                         ",
    "upload_window": "Hochladen",
    "result_window": "Ergebnisse",
  },
  "en": {
    "upload_button": "Upload",
    "main_drag_drop": 'Drag and Drop file here <br> Or <br> Click to Upload',
    "drag_here": "Drag here",
    "drop": "Drop",
    "agb_label": "<b>I Accept the Conditions.</b>",
    "explanation_header": "Facial Palsy grade detection with House-Brackmann",
    "explanation_text": "Test environment of the API connection. <br> \
                         Uploaded images get transfered to the Server. <br> \
                         After the calculation, all images are automatically deleted from the Server. <br> \
                         <br> \
                         This system is intended to provide an independent forecast. \
                         It is a general assessment and not a medical classification! <br> \
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



function updatetags(data) {
  let show = document.createElement("h3");
  $("#output").text(JSON.stringify(data, null, "\t"));

  $("#howto").html(languages[set_lang]["main_drag_drop"])
}






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
