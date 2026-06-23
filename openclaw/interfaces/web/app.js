/**
 * Client-side logic for the OpenClaw job submission form.
 *
 * Performs a lightweight URL shape check (not full YouTube validation —
 * that lives in core/url_validator.py), POSTs to /api/jobs, and renders a
 * minimal success/error message. No polling or live status tracking.
 */
(function () {
  "use strict";

  const form = document.getElementById("job-form");
  const input = document.getElementById("url");
  const submitBtn = document.getElementById("submit-btn");
  const message = document.getElementById("message");

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

  form.addEventListener("submit", async function (event) {
    event.preventDefault();

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

      setMessage("Job queued — ID: " + data.job_id, "success");
      form.reset();
    } catch (err) {
      setMessage("Network error — could not reach the server.", "error");
    } finally {
      setLoading(false);
    }
  });
})();
