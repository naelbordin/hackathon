(() => {
  const chatForm = document.getElementById("chat-form");
  const searching = document.getElementById("searching");
  const toggle = document.getElementById("toggle-less");
  const lessResults = document.getElementById("less-results");
  const actionField = document.getElementById("action-field");
  const skipBtn = document.getElementById("skip-btn");
  const chatFeed = document.getElementById("chat-feed");
  const resumeForm = document.getElementById("resume-form");
  const loading = document.getElementById("loading");

  if (chatFeed) {
    chatFeed.scrollTop = chatFeed.scrollHeight;
  }

  if (chatForm) {
    const step = parseInt(document.body.dataset.step || "0", 10);
    const totalSteps = parseInt(document.body.dataset.totalSteps || "0", 10);
    const role = document.body.dataset.role || "none";
    chatForm.addEventListener("submit", (event) => {
      if (role !== "none" && step === totalSteps - 1) {
        event.preventDefault();
        if (searching) {
          searching.classList.add("active");
        }
        setTimeout(() => chatForm.submit(), 1200);
      }
    });

    const chatInput = chatForm.querySelector("textarea[name=\"query\"]");
    if (chatInput) {
      chatInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
          event.preventDefault();
          chatForm.requestSubmit();
        }
      });
    }
  }

  if (skipBtn && actionField) {
    skipBtn.addEventListener("click", () => {
      actionField.value = "skip";
    });
  }

  if (toggle && lessResults) {
    toggle.addEventListener("click", () => {
      const open = lessResults.classList.toggle("open");
      toggle.textContent = open ? "Masquer les pistes proches" : "Explorer des pistes proches";
    });
  }

  if (resumeForm && loading) {
    resumeForm.addEventListener("submit", () => {
      loading.classList.add("active");
    });
  }
})();
