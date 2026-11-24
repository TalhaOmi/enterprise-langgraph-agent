"use strict";
// Everything the server sends is inserted as plain text (textContent): model output is untrusted and must never run as HTML.
let token = null;
let threadId = null;
const $ = (id) => document.getElementById(id);

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
}

function decodeClaims(t) {
  try {
    const p = t.split(".")[1].replace(/-/g, "+").replace(/_/g, "/");
    return JSON.parse(atob(p));
  } catch (e) { return null; }
}

function bubble(who, text, status) {
  const b = el("div", "bubble " + who);
  if (status) b.appendChild(el("div", "chip " + status, status.replace("_", " ")));
  b.appendChild(el("div", "", text));
  $("log").appendChild(b);
  b.scrollIntoView({ block: "end" });
  return b;
}

async function api(path, method, body) {
  const res = await fetch(path, {
    method,
    headers: { "Content-Type": "application/json", Authorization: "Bearer " + token },
    body: body ? JSON.stringify(body) : undefined,
  });
  let data = {};
  try { data = await res.json(); } catch (e) { /* non-JSON error body */ }
  if (!res.ok) throw new Error(res.status + ": " + (data.detail || res.statusText));
  return data;
}

function approvalCard(approvalId, actions, requestedBy) {
  const card = el("div", "card");
  card.appendChild(el("strong", "", "Approval needed" + (requestedBy ? " (requested by " + requestedBy + ")" : "")));
  card.appendChild(el("div", "muted", JSON.stringify(actions)));
  const row = el("div", "row");
  for (const [label, approve] of [["Approve", true], ["Reject", false]]) {
    const btn = el("button", approve ? "primary" : "", label);
    btn.type = "button";
    btn.addEventListener("click", async () => {
      row.querySelectorAll("button").forEach((b) => (b.disabled = true));
      try {
        const out = await api("/v1/approvals/" + encodeURIComponent(approvalId), "POST", { approve, reason: "" });
        bubble("bot", out.answer || "(no answer)", out.status);
        card.remove();
        loadApprovals();
      } catch (e) {
        bubble("bot", e.message, "error");
        row.querySelectorAll("button").forEach((b) => (b.disabled = false));
      }
    });
    row.appendChild(btn);
  }
  card.appendChild(row);
  return card;
}

async function loadApprovals() {
  const box = $("approvallist");
  box.textContent = "";
  if (!token) return;
  try {
    const items = await api("/v1/approvals", "GET");
    if (!items.length) box.appendChild(el("p", "muted", "Nothing pending."));
    for (const a of items) box.appendChild(approvalCard(a.approval_id, a.actions, a.requested_by));
  } catch (e) {
    box.appendChild(el("p", "muted", e.message.startsWith("403") ? "Your role cannot approve actions." : e.message));
  }
}

$("tokenform").addEventListener("submit", (ev) => {
  ev.preventDefault();
  token = $("token").value.trim();
  $("token").value = "";
  const c = decodeClaims(token);
  $("who").textContent = c ? "Signed in as " + c.sub + " (tenant " + c.tenant + ", roles " + (c.roles || []).join(", ") + ")" : "Token set (could not read details).";
  loadApprovals();
});
$("clear").addEventListener("click", () => { token = null; threadId = null; $("who").textContent = "Not signed in."; $("approvallist").textContent = ""; });
$("refresh").addEventListener("click", loadApprovals);

$("chatform").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  if (!token) { bubble("bot", "Paste an access token first.", "error"); return; }
  const text = $("msg").value.trim();
  if (!text) return;
  $("msg").value = "";
  $("send").disabled = true;
  bubble("user", text);
  try {
    const out = await api("/v1/chat", "POST", { message: text, thread_id: threadId });
    threadId = out.thread_id;
    if (out.status === "pending_approval") {
      const b = bubble("bot", "This action needs a human approver. Share this request with one.", "pending");
      b.appendChild(approvalCard(out.approval_id, out.actions, null));
    } else {
      bubble("bot", out.answer || "(no answer)", out.status);
    }
  } catch (e) {
    bubble("bot", e.message, "error");
  } finally {
    $("send").disabled = false;
    $("msg").focus();
  }
});
