const POLL_MILLISECONDS = 2000;
const UNREACHABLE = "Can't reach the page's server. Is `repolore ui` still running?";

async function text(url, options) {
  const response = await fetch(url, options);
  return response.text();
}

function loadInto(form) {
  const target = document.querySelector(form.dataset.into);
  target.textContent = form.dataset.waiting;
  text(form.action, { method: "POST" })
    .then((html) => {
      target.innerHTML = html;
    })
    .catch(() => {
      target.textContent = UNREACHABLE;
    });
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

for (const form of document.querySelectorAll("form[data-into]")) {
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    loadInto(form);
  });
}

if (document.querySelector("#refresh[data-running]")) {
  setInterval(pollRefresh, POLL_MILLISECONDS);
}
