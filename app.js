const $ = (id) => document.getElementById(id);
const statusLabels = {
  parsing: "正在解析分享链接…",
  downloading: "已找到视频，正在下载并校验…",
  transcribing: "视频已下载，正在准备语音转写…",
  diarizing: "文字已识别，正在区分不同说话人…",
  complete: "文字稿已生成。",
  error: "处理失败。",
};
let activeTask = null;
let pollTimer = null;
let pollFailures = 0;
const savedTaskKey = "wx_transcript_active_task";

function stopPolling() {
  clearTimeout(pollTimer);
  pollTimer = null;
}

function schedulePoll(taskId, delay = 1500) {
  stopPolling();
  if (taskId === activeTask) pollTimer = setTimeout(() => pollTask(taskId), delay);
}

function showError(message = "") {
  $("error").textContent = message;
  $("error").classList.toggle("hidden", !message);
}

function renderStatus(status) {
  $("statusText").textContent = statusLabels[status] || "处理中…";
  const order = ["parsing", "downloading", "transcribing", "diarizing"];
  const current = status === "complete" ? order.length : order.indexOf(status);
  document.querySelectorAll(".step").forEach((el, index) => {
    el.classList.toggle("active", index === current && status !== "complete");
    el.classList.toggle("done", index < current || status === "complete");
  });
}

function timeLabel(seconds) {
  const value = Math.max(0, Math.floor(seconds || 0));
  return `${String(Math.floor(value / 60)).padStart(2, "0")}:${String(value % 60).padStart(2, "0")}`;
}

function renderTask(task) {
  renderStatus(task.status);
  if (task.progress_message) $("statusText").textContent = task.progress_message;
  const meta = task.metadata || {};
  if (meta.title || task.status === "complete") {
    $("result").classList.remove("hidden");
    $("cover").src = meta.cover_url || "";
    $("cover").classList.toggle("hidden", !meta.cover_url);
    $("title").textContent = meta.title || "视频来源";
    $("author").textContent = meta.author ? `作者：${meta.author}` : "作者：未知";
    $("description").textContent = meta.description || "暂无描述";
    $("mediaInfo").textContent = [meta.quality, task.media_bytes ? `${(task.media_bytes / 1024 / 1024).toFixed(2)} MB` : ""].filter(Boolean).join(" · ");
    $("sourceLink").href = meta.source_url || task.source_url;
  }
  if (task.status === "complete") {
    $("transcriptLines").replaceChildren();
    if (task.segments?.length) {
      for (const segment of task.segments) {
        const row = document.createElement("p");
        row.className = "transcript-line";
        const stamp = document.createElement("span");
        stamp.className = "timestamp";
        stamp.textContent = timeLabel(segment.start);
        const words = document.createElement("span");
        const speaker = document.createElement("strong");
        speaker.className = "speaker-label";
        speaker.textContent = segment.speaker || "说话人 A";
        words.textContent = segment.text;
        row.append(stamp, speaker, words);
        $("transcriptLines").append(row);
      }
      const languageNames = {auto: "自动检测", zh: "中文", en: "英语", ja: "日语", ko: "韩语"};
      const language = languageNames[task.language] || task.language || "自动检测";
      const speakerCount = Number.isInteger(task.speaker_count) ? task.speaker_count : 1;
      const speakerSummary = speakerCount ? `模型标注 ${speakerCount} 位说话人` : "未检测到可确认的说话人";
      const engineName = task.engine === "aliyun-refined" ? `阿里云 ${task.provider_model || "Paraformer"} + 独立声纹校正` : task.engine === "aliyun" ? `阿里云 ${task.provider_model || "Paraformer"}` : "本地 Whisper + 声纹";
      const details = `${engineName} · 识别语言：${language} · ${speakerSummary} · 共 ${task.segments.length} 段。`;
      const visual = task.visual_evidence;
      const visualStats = visual?.active_speaker_available
        ? `连续人脸轨迹 ${visual.track_count || 0} 段，主动说话候选 ${visual.active_windows?.length || 0} 段`
        : "";
      const visualHint = visual?.reason ? `画面证据：${visual.reason}${visualStats ? `（${visualStats}）` : ""}` : "";
      const fusionHint = task.visual_fusion?.reason ? ` ${task.visual_fusion.reason}` : "";
      const textReviewHint = task.text_review?.reason ? ` ${task.text_review.reason}` : "";
      $("transcriptHint").textContent = [details, task.speaker_notice, visualHint + fusionHint + textReviewHint].filter(Boolean).join(" ");
    } else {
      $("transcriptHint").textContent = "视频里没有识别出清晰语音。";
    }
    $("downloadText").href = `/api/transcript/${task.task_id}.txt`;
    $("downloadText").classList.remove("hidden");
  }
  if (task.status === "error") showError(task.error || "处理失败，请重试。");
}

async function pollTask(taskId) {
  try {
    const response = await fetch(`/api/task?id=${encodeURIComponent(taskId)}`, {cache: "no-store"});
    const task = await response.json();
    if (response.status === 404 && taskId === activeTask) {
      stopPolling();
      sessionStorage.removeItem(savedTaskKey);
      activeTask = null;
      $("analyzeBtn").disabled = false;
      $("analyzeBtn").textContent = "生成文字稿";
      showError("原任务已不存在（服务重启后任务进度不会保留），请重新点击“生成文字稿”。");
      return;
    }
    if (!response.ok) throw new Error(task.error || "读取任务失败");
    if (taskId !== activeTask) return;
    pollFailures = 0;
    showError();
    renderTask(task);
    if (task.status === "complete" || task.status === "error") {
      stopPolling();
      sessionStorage.removeItem(savedTaskKey);
      $("analyzeBtn").disabled = false;
      $("analyzeBtn").textContent = "生成文字稿";
    } else {
      schedulePoll(taskId);
    }
  } catch (error) {
    if (taskId !== activeTask) return;
    pollFailures++;
    if (pollFailures >= 3) showError(`暂时无法读取进度，正在自动重试（${error.message}）。任务可能仍在后台运行。`);
    schedulePoll(taskId, Math.min(1500 * 2 ** Math.min(pollFailures, 4), 15000));
  }
}

async function analyze() {
  const url = $("shareUrl").value.trim();
  showError();
  if (!url) return showError("请先粘贴视频号分享链接。");
  stopPolling();
  activeTask = null;
  sessionStorage.removeItem(savedTaskKey);
  pollFailures = 0;
  $("analyzeBtn").disabled = true;
  $("analyzeBtn").textContent = "处理中…";
  $("progressCard").classList.remove("hidden");
  $("result").classList.add("hidden");
  $("downloadText").classList.add("hidden");
  $("transcriptLines").replaceChildren();
  $("transcriptHint").textContent = "等待转写结果…";
  renderStatus("parsing");
  try {
    const numSpeakers = Number.parseInt($("speakerCount").value, 10);
    const response = await fetch("/api/analyze", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({url, num_speakers: numSpeakers, engine: $("engine").value})});
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error(data.error || "创建任务失败");
    activeTask = data.task_id;
    sessionStorage.setItem(savedTaskKey, activeTask);
    await pollTask(activeTask);
  } catch (error) {
    showError(error.message);
    $("analyzeBtn").disabled = false;
    $("analyzeBtn").textContent = "生成文字稿";
  }
}

$("analyzeBtn").addEventListener("click", analyze);
$("shareUrl").addEventListener("keydown", (event) => {if (event.key === "Enter") analyze();});
const savedTask = sessionStorage.getItem(savedTaskKey);
if (savedTask && /^[0-9a-f]{32}$/.test(savedTask)) {
  activeTask = savedTask;
  $("analyzeBtn").disabled = true;
  $("analyzeBtn").textContent = "处理中…";
  $("progressCard").classList.remove("hidden");
  pollTask(savedTask);
}
fetch("/api/health", {cache: "no-store"}).then(r => r.json()).then(data => {
  if (!activeTask && data.default_engine) $("engine").value = data.default_engine;
}).catch(() => {});
