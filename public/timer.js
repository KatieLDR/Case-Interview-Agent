(function () {
  // ── Config ──────────────────────────────────────────────────────────────
  const DURATION_SECONDS = 30; // 30 minutes(change to 30*60)
  const TRIGGER_TEXT     = "Your 30-minute session has started";
  const END_TEXT         = "Session Ended";

  // ── State ───────────────────────────────────────────────────────────────
  let timerStarted = false;
  let intervalId   = null;
  let secondsLeft  = DURATION_SECONDS;
  let overlay      = null;

  // ── DOM helpers ─────────────────────────────────────────────────────────

  function createOverlay() {
    const el = document.createElement("div");
    el.id = "ba-timer-overlay";
    el.style.cssText = [
      "position: fixed",
      "top: 50%",
      "left: 16px",
      "transform: translateY(-50%)",
      "z-index: 9999",
      "background: rgba(30, 30, 30, 0.90)",
      "color: #ffffff",
      "font-family: monospace",
      "font-size: 22px",
      "font-weight: bold",
      "padding: 14px 20px",
      "border-radius: 10px",
      "pointer-events: none",
      "user-select: none",
      "box-shadow: 0 4px 16px rgba(0,0,0,0.5)",
      "letter-spacing: 1px",
    ].join("; ");
    document.body.appendChild(el);
    return el;
  }

  function createPopup() {
    const backdrop = document.createElement("div");
    backdrop.id = "ba-timer-popup-backdrop";
    backdrop.style.cssText = [
      "position: fixed",
      "inset: 0",
      "z-index: 10000",
      "background: rgba(0,0,0,0.55)",
      "display: flex",
      "align-items: center",
      "justify-content: center",
    ].join("; ");

    const box = document.createElement("div");
    box.style.cssText = [
      "background: #ffffff",
      "color: #111111",
      "border-radius: 12px",
      "padding: 28px 32px",
      "max-width: 400px",
      "width: 90%",
      "text-align: center",
      "box-shadow: 0 8px 32px rgba(0,0,0,0.3)",
      "font-family: sans-serif",
    ].join("; ");

    const title = document.createElement("p");
    title.style.cssText = "font-size: 20px; font-weight: 700; margin: 0 0 12px 0;";
    title.textContent = "⏰ Time's up!";

    const msg = document.createElement("p");
    msg.style.cssText = "font-size: 15px; line-height: 1.6; margin: 0 0 20px 0;";
    msg.textContent =
      "Please click the 📊 Get Summary & End Session button, " +
      "then copy your Session ID into the survey.";

    const btn = document.createElement("button");
    btn.textContent = "Got it — close this";
    btn.style.cssText = [
      "background: #111111",
      "color: #ffffff",
      "border: none",
      "border-radius: 8px",
      "padding: 10px 20px",
      "font-size: 14px",
      "cursor: pointer",
      "font-family: sans-serif",
    ].join("; ");
    btn.onclick = function () {
      backdrop.remove();
    };

    box.appendChild(title);
    box.appendChild(msg);
    box.appendChild(btn);
    backdrop.appendChild(box);
    document.body.appendChild(backdrop);
  }

  function formatTime(seconds) {
    const m = Math.floor(seconds / 60).toString().padStart(2, "0");
    const s = (seconds % 60).toString().padStart(2, "0");
    return `⏱ ${m}:${s}`;
  }

  function stopTimer() {
    if (intervalId) {
      clearInterval(intervalId);
      intervalId = null;
    }
    if (overlay) {
      overlay.style.display = "none";
    }
    console.log("[BA Timer] stopped — session ended");
  }

  // ── Timer logic ─────────────────────────────────────────────────────────

  function startTimer() {
    if (timerStarted) return;
    timerStarted = true;

    overlay = createOverlay();
    overlay.textContent = formatTime(secondsLeft);

    intervalId = setInterval(() => {
      secondsLeft -= 1;
      overlay.textContent = formatTime(secondsLeft);

      if (secondsLeft <= 0) {
        clearInterval(intervalId);
        overlay.textContent = "⏱ 00:00";
        createPopup();
      }
    }, 1000);

    console.log("[BA Timer] started — 30 minutes");
  }

  // ── MutationObserver ────────────────────────────────────────────────────

  const observer = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      for (const node of mutation.addedNodes) {
        if (node.nodeType !== Node.ELEMENT_NODE) continue;
        const text = node.textContent || "";

        if (!timerStarted && text.includes(TRIGGER_TEXT)) {
          startTimer();
        }

        if (timerStarted && text.includes(END_TEXT)) {
          stopTimer();
        }
      }
    }
  });

  function init() {
    observer.observe(document.body, { childList: true, subtree: true });
    console.log("[BA Timer] observer active");
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();