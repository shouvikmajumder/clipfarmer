(function () {
  "use strict";

  const content = document.getElementById("results-content");

  function jobIdFromPath() {
    // path is /jobs/<job_id>
    const parts = window.location.pathname.split("/").filter(Boolean);
    return parts[1] || null;
  }

  function render(job) {
    const title = job.video_title || "Untitled video";
    const clipCount = job.clip_count || 0;
    const jobId = job.job_id;

    const clipMsg = clipCount > 0
      ? clipCount + " clip" + (clipCount === 1 ? "" : "s") + " detected"
      : "No clips detected — full video below";

    content.innerHTML =
      '<div class="video-card">' +
        '<h2 class="video-title">' + escapeHtml(title) + "</h2>" +
        '<p class="clip-count">' + escapeHtml(clipMsg) + "</p>" +
        '<video class="video-player" controls preload="metadata">' +
          '<source src="/jobs/' + jobId + '/video" type="video/mp4" />' +
          "Your browser does not support the video tag." +
        "</video>" +
        '<a class="download-link" href="/jobs/' + jobId + '/video" download>Download video</a>' +
      "</div>";
  }

  function renderError(msg) {
    content.innerHTML = '<p class="job-error">' + escapeHtml(msg) + "</p>";
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  async function pollUntilDone(jobId) {
    const TERMINAL = new Set(["complete", "failed", "cancelled"]);
    while (true) {
      try {
        const res = await fetch("/api/jobs/" + jobId);
        if (!res.ok) {
          renderError("Could not load job (HTTP " + res.status + ").");
          return;
        }
        const data = await res.json();
        if (TERMINAL.has(data.state)) {
          if (data.state === "complete") {
            render(data);
          } else {
            renderError("Job " + data.state + (data.error ? ": " + data.error : "."));
          }
          return;
        }
        // still running — update label
        const LABELS = { queued: "Queued…", downloading: "Downloading video…", detecting: "Detecting clips…" };
        content.innerHTML = '<p class="status-label">' + (LABELS[data.state] || data.state) + "</p>";
      } catch (_) {
        // network hiccup
      }
      await new Promise(function (r) { setTimeout(r, 2000); });
    }
  }

  async function init() {
    const jobId = jobIdFromPath();
    if (!jobId) {
      renderError("No job ID in URL.");
      return;
    }

    try {
      const res = await fetch("/api/jobs/" + jobId);
      if (!res.ok) {
        renderError("Job not found.");
        return;
      }
      const data = await res.json();
      if (data.state === "complete") {
        render(data);
      } else if (data.state === "failed" || data.state === "cancelled") {
        renderError("Job " + data.state + (data.error ? ": " + data.error : "."));
      } else {
        await pollUntilDone(jobId);
      }
    } catch (_) {
      renderError("Network error — could not reach the server.");
    }
  }

  init();
})();
