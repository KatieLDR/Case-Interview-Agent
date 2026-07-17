(function () {
  // ── Config ──────────────────────────────────────────────────────────────
  const DURATION_SECONDS = 15* 60;
  const TRIGGER_TEXT     = "Your 15-minute session has started";
  const END_TEXT         = "Session Ended";

  // ── State ───────────────────────────────────────────────────────────────
  let timerStarted = false;
  let intervalId   = null;
  let secondsLeft  = DURATION_SECONDS;
  let overlay      = null;

  // ── Session Guard ────────────────────────────────────────────────────────

  function beforeUnloadHandler(e) {
    e.preventDefault();
    e.returnValue = ""; // Required for Chrome to trigger the native dialog
  }

  function activateGuard() {
    window.addEventListener("beforeunload", beforeUnloadHandler);
    console.log("[BA Guard] session guard active");
  }

  function deactivateGuard() {
    window.removeEventListener("beforeunload", beforeUnloadHandler);
    console.log("[BA Guard] session guard removed — session ended");
  }

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
      "Times up! Please click the ‼️End Session, " +
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

    console.log("[BA Timer] started — 15 minutes");
  }

  // ── Qualtrics / Prolific linkage ──────────────────────────────────────────
  // Qualtrics appends ?qrid=<ResponseID> when redirecting to the agent link.
  // For the g2 (Prolific) group the redirect also carries PROLIFIC_PID / STUDY_ID
  // / SESSION_ID (piped from Qualtrics embedded data, originally set by Prolific).
  // We forward them to Python via the Chainlit window_message channel so they get
  // stored on the Firestore session doc.
  // Analysis join: Qualtrics ResponseID = Firestore session.participant_id;
  //                Prolific PID = Firestore session.prolific_pid.
  //
  // window.postMessage fires before Chainlit's React listener is registered, so
  // we retry at 1 s and 3 s — cl.user_session.set is idempotent for same value.
  // Param names are case-sensitive and must match Qualtrics/Prolific exactly.
  const _params = new URLSearchParams(window.location.search);
  const qrid = _params.get("qrid");
  const prolificPid = _params.get("PROLIFIC_PID");
  const studyId = _params.get("STUDY_ID");
  const prolificSessionId = _params.get("SESSION_ID");
  if (qrid || prolificPid || studyId || prolificSessionId) {
    if (qrid) sessionStorage.setItem("ba_qrid", qrid);
    function _sendPid() {
      window.postMessage({
        type: "ba_pid",
        value: qrid,
        prolific_pid: prolificPid,
        study_id: studyId,
        prolific_session_id: prolificSessionId,
      }, "*");
    }
    _sendPid();
    setTimeout(_sendPid, 1000);
    setTimeout(_sendPid, 3000);
    console.log("[BA Qualtrics] ids forwarded to backend:",
      { qrid, prolificPid, studyId, prolificSessionId });
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

        if (text.includes(END_TEXT)) {
          stopTimer();
          deactivateGuard();
        }
      }
    }
  });

  // ── Disable browser translation ──────────────────────────────────────────
  // Browser auto-translation (notably Edge's built-in translator) rewrites
  // React's text nodes, which crashes Chainlit's frontend with
  // "Failed to execute 'insertBefore' on 'Node' ... not a child of this node"
  // during token streaming. Marking the page notranslate prevents it.
  function disableTranslation() {
    document.documentElement.setAttribute("lang", "en"); // no mismatch → no auto-translate prompt
    document.documentElement.setAttribute("translate", "no");
    document.documentElement.classList.add("notranslate");
    document.body.classList.add("notranslate");
    const meta = document.createElement("meta");
    meta.name = "google";
    meta.content = "notranslate";
    document.head.appendChild(meta);
    console.log("[BA Timer] page marked notranslate");
  }

  function init() {
    disableTranslation();
    observer.observe(document.body, { childList: true, subtree: true });
    console.log("[BA Timer] observer active");
    activateGuard();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();