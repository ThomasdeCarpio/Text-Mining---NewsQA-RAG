(function () {
  var slides = Array.prototype.slice.call(document.querySelectorAll(".slide"));
  var counter = document.getElementById("slide-counter");
  var notesDrawer = document.getElementById("notes-drawer");
  var notesBody = document.getElementById("notes-body");
  var current = 0;

  function render() {
    slides.forEach(function (s, i) {
      s.classList.toggle("active", i === current);
    });
    if (counter) counter.textContent = (current + 1) + " / " + slides.length;
    if (notesDrawer && notesDrawer.classList.contains("open")) {
      renderNotes();
    }
  }

  function renderNotes() {
    var aside = slides[current].querySelector(".notes");
    notesBody.innerHTML = aside ? aside.innerHTML : "";
  }

  function go(delta) {
    current = Math.min(Math.max(current + delta, 0), slides.length - 1);
    render();
  }

  function toggleNotes() {
    notesDrawer.classList.toggle("open");
    if (notesDrawer.classList.contains("open")) renderNotes();
  }

  document.addEventListener("keydown", function (e) {
    if (e.key === "ArrowRight" || e.key === "PageDown" || e.key === " ") go(1);
    else if (e.key === "ArrowLeft" || e.key === "PageUp") go(-1);
    else if (e.key === "n" || e.key === "N") toggleNotes();
    else if (e.key === "Home") { current = 0; render(); }
    else if (e.key === "End") { current = slides.length - 1; render(); }
  });

  var prevBtn = document.getElementById("btn-prev");
  var nextBtn = document.getElementById("btn-next");
  var notesBtn = document.getElementById("btn-notes");
  if (prevBtn) prevBtn.addEventListener("click", function () { go(-1); });
  if (nextBtn) nextBtn.addEventListener("click", function () { go(1); });
  if (notesBtn) notesBtn.addEventListener("click", toggleNotes);

  render();
})();
