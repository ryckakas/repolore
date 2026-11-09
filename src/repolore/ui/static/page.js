const POLL_MILLISECONDS = 2000;
const COPIED_MILLISECONDS = 1500;
const UNREACHABLE = "Can't reach the page's server. Is `repolore ui` still running?";

async function text(url, options) {
  const response = await fetch(url, options);
  return response.text();
}

function fill(target, url, options) {
  text(url, options)
    .then((html) => {
      target.innerHTML = html;
    })
    .catch(() => {
      target.textContent = UNREACHABLE;
    });
}

function submitInto(form) {
  const target = document.querySelector(form.dataset.into);
  target.textContent = form.dataset.waiting;
  fill(target, form.action, { method: "POST", body: new URLSearchParams(new FormData(form)) });
}

async function pollRefresh() {
  let html;
  try {
    html = await text("/refresh");
  } catch {
    return;
  }
  const holder = document.createElement("div");
  holder.innerHTML = html;
  if (!holder.firstElementChild) {
    return;
  }
  if (!holder.firstElementChild.hasAttribute("data-running")) {
    location.reload();
    return;
  }
  document.querySelector("#refresh").replaceWith(holder.firstElementChild);
}

function press(buttons, chosen) {
  for (const button of buttons) {
    button.setAttribute("aria-pressed", String(button === chosen));
  }
}

function showTool(button) {
  const name = button.dataset.tool;
  press(document.querySelectorAll("[data-tool]"), button);
  for (const panel of document.querySelectorAll("[data-tool-panel]")) {
    panel.hidden = panel.dataset.toolPanel !== name;
  }
  document.querySelector("#result").replaceChildren();
}

function showView(button) {
  const name = button.dataset.view;
  press(document.querySelectorAll("[data-view]"), button);
  for (const panel of document.querySelectorAll("[data-view-panel]")) {
    panel.hidden = panel.dataset.viewPanel !== name;
  }
}

function readAll(button) {
  const url = button.dataset.read;
  showTool(document.querySelector('[data-tool="read"]'));
  const form = document.querySelector('form[action="/tools/read"]');
  form.reset();
  form.elements.url.value = url;
  form.requestSubmit();
}

function copy(button) {
  const label = button.textContent;
  const content = button.closest(".copyable").querySelector("pre").textContent;
  button.disabled = true;
  navigator.clipboard
    .writeText(content)
    .then(() => {
      button.textContent = "Copied";
    })
    .catch(() => {
      button.textContent = "Couldn't copy";
    })
    .finally(() => {
      setTimeout(() => {
        button.textContent = label;
        button.disabled = false;
      }, COPIED_MILLISECONDS);
    });
}

function filterRepos(input) {
  const wanted = input.value.trim().toLowerCase();
  for (const row of input.form.querySelectorAll(".repos li")) {
    row.hidden = !row.textContent.toLowerCase().includes(wanted);
  }
}

function showTicked(form) {
  const ticked = form.querySelectorAll('[name="repo"]:checked').length;
  const count = form.elements.every.checked ? "All" : ticked;
  form.querySelector("[data-ticked]").textContent = `${count} ticked`;
}

const CLICKS = {
  "[data-tool]": showTool,
  "[data-view]": showView,
  "[data-read]": readAll,
  "[data-copy]": copy,
};

document.addEventListener("click", (event) => {
  for (const [selector, act] of Object.entries(CLICKS)) {
    const button = event.target.closest(selector);
    if (button) {
      act(button);
    }
  }
});

document.addEventListener("submit", (event) => {
  if (event.target.matches("form[data-into]")) {
    event.preventDefault();
    submitInto(event.target);
  }
});

document.addEventListener("input", (event) => {
  if (event.target.matches("[data-filter-repos]")) {
    filterRepos(event.target);
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && event.target.matches("[data-filter-repos]")) {
    event.preventDefault();
  }
});

document.addEventListener("change", (event) => {
  if (event.target.matches('input[type="checkbox"][name="repo"], [name="every"]')) {
    showTicked(event.target.form);
  }
});

for (const target of document.querySelectorAll("[data-load]")) {
  fill(target, target.dataset.load);
}

if (document.querySelector("#refresh[data-running]")) {
  setInterval(pollRefresh, POLL_MILLISECONDS);
}
