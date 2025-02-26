let menu = document.querySelector("#menu-btn");
let navbar = document.querySelector(".navbar");

menu.onclick = () => {
  menu.classList.toggle("fa-times");
  navbar.classList.toggle("active");
};

window.onscroll = () => {
  menu.classList.remove("fa-times");
  navbar.classList.remove("active");
};
function closePopup() {
    document.getElementById("customPopup").style.display = "none";
}

window.onload = function() {
    setTimeout(() => {
        document.getElementById("customPopup").style.display = "flex";
    }, 500); // Delay popup by 0.5s
};

