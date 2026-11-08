const POLL_MILLISECONDS = 2000;
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

document.addEventListener("submit", (event) => {
  if (event.target.matches("form[data-into]")) {
    event.preventDefault();
    submitInto(event.target);
  }
});

for (const target of document.querySelectorAll("[data-load]")) {
  fill(target, target.dataset.load);
}

if (document.querySelector("#refresh[data-running]")) {
  setInterval(pollRefresh, POLL_MILLISECONDS);
}
