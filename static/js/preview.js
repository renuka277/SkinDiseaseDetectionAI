// Image Preview Inside Upload Box

const imageInput = document.getElementById("image");
const preview = document.getElementById("preview");
const uploadContent = document.getElementById("uploadContent");

imageInput.addEventListener("change", function () {

    const file = this.files[0];

    if (!file) {

        preview.src = "";
        preview.style.display = "none";
        uploadContent.style.display = "block";

        return;
    }

    const reader = new FileReader();

    reader.onload = function (e) {

        preview.src = e.target.result;

        preview.style.display = "block";

        uploadContent.style.display = "none";

    };

    reader.readAsDataURL(file);

});


// Loading Animation

const form = document.getElementById("uploadForm");

form.addEventListener("submit", function (e) {

    if (imageInput.files.length === 0) {

        alert("Please select an image first.");

        e.preventDefault();

        return;
    }

    document.getElementById("loading").style.display = "flex";

});