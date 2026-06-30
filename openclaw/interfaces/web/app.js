/**
 * Client-side logic for the clipfarmer.bot job submission form.
 *
 * Performs a lightweight URL shape check (not full YouTube validation —
 * that lives in core/url_validator.py), POSTs to /api/jobs, then polls
 * GET /api/jobs/<id> every 2 s to update the progress bar and status label.
 */
(function () {
  "use strict";

  const form = document.getElementById("job-form");
  const input = document.getElementById("url");
  const submitBtn = document.getElementById("submit-btn");
  const message = document.getElementById("message");
  const progressWrapper = document.getElementById("progress-wrapper");
  const progressBar = document.getElementById("progress-bar");
  const statusLabel = document.getElementById("status-label");
  const resultLink = document.getElementById("result-link");

  const STAGE_PROGRESS = { queued: 5, downloading: 33, detecting: 66, complete: 100 };
  const STAGE_LABELS = {
    queued: "Queued…",
    downloading: "Downloading video…",
    detecting: "Detecting clips…",
    complete: "Done!",
    failed: "Failed.",
    cancelled: "Cancelled.",
  };

  let pollTimer = null;

  function setMessage(text, kind) {
    message.textContent = text;
    message.classList.remove("error", "success");
    if (kind) {
      message.classList.add(kind);
    }
  }

  function looksLikeUrl(value) {
    try {
      const parsed = new URL(value);
      return parsed.protocol === "http:" || parsed.protocol === "https:";
    } catch {
      return false;
    }
  }

  function setLoading(isLoading) {
    submitBtn.disabled = isLoading;
    submitBtn.textContent = isLoading ? "Processing…" : "Process";
  }

  function setProgress(pct, label) {
    progressBar.style.width = pct + "%";
    statusLabel.textContent = label;
  }

  function startPolling(jobId) {
    progressWrapper.hidden = false;
    setProgress(5, STAGE_LABELS.queued);

    pollTimer = setInterval(async function () {
      try {
        const res = await fetch("/api/jobs/" + jobId);
        if (!res.ok) return;
        const data = await res.json();

        const pct = STAGE_PROGRESS[data.state] ?? 0;
        const label = STAGE_LABELS[data.state] ?? data.state;
        setProgress(pct, label);

        if (data.state === "complete") {
          clearInterval(pollTimer);
          pollTimer = null;
          if (data.clip_count > 0) {
            resultLink.href = "/jobs/" + jobId;
            resultLink.hidden = false;
          } else {
            setMessage("Processing complete — no clips were detected.", "success");
          }
        } else if (data.state === "failed" || data.state === "cancelled") {
          clearInterval(pollTimer);
          pollTimer = null;
          setMessage(data.error || "Job " + data.state + ".", "error");
        }
      } catch (_) {
        // network hiccup — keep polling
      }
    }, 2000);
  }

  form.addEventListener("submit", async function (event) {
    event.preventDefault();

    // Reset progress UI so re-submitting a new URL starts fresh
    if (pollTimer !== null) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
    progressWrapper.hidden = true;
    progressBar.style.width = "0%";
    resultLink.hidden = true;

    const url = input.value.trim();

    if (!url) {
      setMessage("Please paste a YouTube URL.", "error");
      input.focus();
      return;
    }

    if (!looksLikeUrl(url)) {
      setMessage("That doesn't look like a valid URL.", "error");
      input.focus();
      return;
    }

    setLoading(true);
    setMessage("");

    try {
      const response = await fetch("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: url }),
      });

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        setMessage(data.error || "Something went wrong. Please try again.", "error");
        return;
      }

      startPolling(data.job_id);
      form.reset();
    } catch (err) {
      setMessage("Network error — could not reach the server.", "error");
    } finally {
      setLoading(false);
    }
  });
})();
