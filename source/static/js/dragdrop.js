import { dragdrop } from './imagehelper.js'
//idea from https://github.com/shinokada/fastapi-drag-and-drop
$(function () {
  dragdrop();

  function preparedata (file) {
    let data = { 'winWidth': 1, 'imgWidth': 2, 'imgHeight': 3 };

    let jdata = JSON.stringify(data);
	console.log('-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-')
	console.log("Preparing ...")
	console.log(file)

	var formData = new FormData();

	for(var x=0; x < file.length; x++){
		formData.append("files", file[x]);
	}

    uploadData(formData);
  }





  function getFilesFromWebkitDataTransferItems(dataTransferItems) {
  function traverseFileTreePromise(item, path='') {
    return new Promise( resolve => {
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



  // Drop
  $('.upload-area').on('drop', function (e) {
    e.stopPropagation();
    e.preventDefault();
    $("#howto").text("We are uploading your file.");
    let file = e.originalEvent.dataTransfer.files;



	var items = event.dataTransfer.items;
	getFilesFromWebkitDataTransferItems(items).then(files => {
	  for(var i=0; i < files.length; i++){
		  files[i] =new File([new Blob()], files[i].filepath);
	  }
	  console.log("File uploaded: ", files);

	  preparedata(files);
    })

    console.log("done drop.")
  });


  // Open file selector on div click
  $("#uploadfile").click(function () {
    $("#file").click();
  });

  // file selected
  $("#file").change(function () {
	console.log($('#file')[0].files);

    $("#howto").text("Uploading your file.");
    preparedata($('#file')[0].files);
  });
});



// Sending AJAX request and upload file
function uploadData (formdata) {

  $.ajax({
    url: '/upload/new/',
    type: 'post',
    data: formdata,
    contentType: false,
    processData: false,
    success: function (data) {
		console.log(data)
        updatetags(data);
    }
  });
}

function updatetags (data) {
  let show = document.createElement("h3");
  $("#output").text(data.result);

  $("#howto").html("Drag and Drop file here<br />Or<br />Click to Upload")
}
