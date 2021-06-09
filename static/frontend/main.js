let model;

const modelURL = 'http://localhost:5000/model';

const preview = document.getElementById("preview");
const predictButton = document.getElementById("predict");
const clearButton = document.getElementById("clear");
const numberOfFiles = document.getElementById("number-of-files");
const fileInput = document.getElementById('file');

const predict1 = async (modelURL) => {
    if (!model) model = await tf.loadModel(modelURL);
    const files = fileInput.files;

    [...files].map(async (img) => {
        const data = new FormData();
        data.append('file', img);

        const processedImage = await fetch("/api/prepare",
            {
                method: 'POST',
                body: data
            }).then(response => {
                return response.json();
            }).then(result => {
				
                return tf.tensor(result['images']);
            });
      
		
        // shape has to be the same as it was for training of the model
        const prediction = model.predict(tf.reshape(processedImage, shape = [1, 48, 48,3]));
		
       var label1 = prediction.argMax(axis = 1).get([0]);
	  const d = prediction.buffer().get([label1]);
	  var t = d*100
	   if(label1==1){
		   
		   label1="nam";
	   }else{label1="nu";}
        
        renderImageLabel(img,label1,t,processedImage);
    })
};

const renderImageLabel = (img, label1,t,processedImage) => {
    const reader = new FileReader();
    reader.onload = () => {
        preview.innerHTML += `<div class="image-block">
                                     
									 <img src="${reader.result}" class="image-block_loaded" id="source"/>
	
	<div >${label1}</div>
	<div >${t}%</div>

                                   
                                       
                              </div>`;

    };
    reader.readAsDataURL(img);
   
};


fileInput.addEventListener("change", () => numberOfFiles.innerHTML = "Selected " + fileInput.files.length + " files", false);
predictButton.addEventListener("click", () => predict1(modelURL));
clearButton.addEventListener("click", () => preview.innerHTML = "");
