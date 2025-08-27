/**
 * API Taixiu SIÊU VIP – by Tele@idol_vannhat
 * Node >= 18 (có fetch sẵn)
 * Endpoint chính: GET /api/custom
 * Trả về:
 * {
 *   id: "Tele@idol_vannhat",
 *   Phien_truoc, Phien_sau,
 *   Xuc_xac: [x1,x2,x3],
 *   Tong, Ket_qua, Du_doan, Do_tin_cay, Giai_thich, Mau_cau
 * }
 */

import express from "express";

const app = express();
const PORT = process.env.PORT || 3000;

// API nguồn (trả lịch sử nhiều phiên, phần tử [0] thường là mới nhất)
const SOURCE_API = "https://sunai.onrender.com/api/taixiu/history";

// ====== Bộ nhớ tạm thời trên server (in-memory) ======
let history = [];            // [{session, dice:[a,b,c], total, result}]
let patternMemory = {};      // n-gram cho chuỗi 'T'/'X' -> đếm tần suất tiếp theo
let modelPredictions = {};   // lưu dự đoán theo session để chấm hiệu năng
const MAX_HISTORY = 300;

// ====== Tiện ích ======
const r01 = (x) => Math.round(x * 100) / 100;
const last = (arr, k = 1) => arr.slice(-k);
const toTX = (res) => (res === "Tài" ? "T" : "X");

// ====== Nạp dữ liệu từ API nguồn & chuẩn hóa ======
async function loadSource() {
  const resp = await fetch(SOURCE_API, { cache: "no-store" });
  const data = await resp.json();

  // Kỳ vọng data là mảng các bản ghi có {session, dice, total, result}
  // Nếu API trả khác, bạn có thể chỉnh map ở đây.
  if (!Array.isArray(data) || data.length === 0) {
    throw new Error("API nguồn không trả mảng dữ liệu hợp lệ");
  }

  // Chuẩn hóa + sort theo session tăng dần
  const norm = data
    .filter(x => x && typeof x.session !== "undefined")
    .map(x => ({
      session: Number(x.session),
      dice: Array.isArray(x.dice) ? x.dice.map(Number) : [0, 0, 0],
      total: Number(x.total ?? 0),
      result: x.result === "Tài" || x.result === "Xỉu"
        ? x.result
        : (Number(x.total ?? 0) >= 11 ? "Tài" : "Xỉu")
    }))
    .sort((a, b) => a.session - b.session);

  // Gộp vào history, tránh trùng lặp
  const seen = new Set(history.map(h => h.session));
  for (const row of norm) {
    if (!seen.has(row.session)) history.push(row);
  }
  if (history.length > MAX_HISTORY) {
    history = history.slice(-MAX_HISTORY);
  }

  // Cập nhật patternMemory từ chuỗi Mau_cau
  rebuildPatternMemory();
}

// ====== Xây dựng bộ nhớ mẫu cầu (n-gram) ======
function rebuildPatternMemory() {
  patternMemory = {};
  const seq = history.map(h => toTX(h.result)); // chuỗi 'T'/'X'
  const N = seq.length;

  // Dùng n-gram với n = 3,4,5 để bắt mẫu cầu dài/ngắn
  const ns = [3, 4, 5];
  for (const n of ns) {
    for (let i = 0; i <= N - (n + 1); i++) {
      const key = seq.slice(i, i + n).join("");      // n ký tự
      const nxt = seq[i + n];                        // ký tự tiếp theo
      if (!patternMemory[n]) patternMemory[n] = {};
      if (!patternMemory[n][key]) patternMemory[n][key] = { T: 0, X: 0 };
      patternMemory[n][key][nxt]++;
    }
  }
}

// ====== Các module phân tích/thuật toán ======
function detectStreakAndBreak(hist) {
  if (hist.length === 0) return { streak: 0, current: null, breakProb: 0 };

  let streak = 1;
  const current = hist[hist.length - 1].result;
  for (let i = hist.length - 2; i >= 0; i--) {
    if (hist[i].result === current) streak++;
    else break;
  }

  const last20 = hist.slice(-20).map(x => x.result);
  const switches = last20.slice(1).reduce((c, r, i) => c + (r !== last20[i] ? 1 : 0), 0);
  const taiCount = last20.filter(r => r === "Tài").length;
  const xiuCount = last20.length - taiCount;
  const imb = last20.length ? Math.abs(taiCount - xiuCount) / last20.length : 0;

  let breakProb = 0;
  if (streak >= 9) breakProb = Math.min(0.7 + switches / 20 + imb * 0.2, 0.95);
  else if (streak >= 6) breakProb = Math.min(0.45 + switches / 15 + imb * 0.3, 0.9);
  else if (streak >= 4 && switches >= 9) breakProb = 0.4;

  // entropy đơn giản
  const pT = taiCount / (last20.length || 1);
  const pX = 1 - pT;
  const entropy = -(pT ? pT * Math.log2(pT) : 0) - (pX ? pX * Math.log2(pX) : 0);
  if (entropy < 0.8) breakProb += 0.1;

  return { streak, current, breakProb: Math.min(breakProb, 1) };
}

function markovTransitionProb(hist) {
  if (hist.length < 5) return { taiProb: 0.5, xiuProb: 0.5 };
  const trans = { Tài: { Tài: 0, Xỉu: 0 }, Xỉu: { Tài: 0, Xỉu: 0 } };
  for (let i = 1; i < hist.length; i++) {
    trans[hist[i - 1].result][hist[i].result]++;
  }
  const cur = hist[hist.length - 1].result;
  const tot = trans[cur].Tài + trans[cur].Xỉu;
  if (!tot) return { taiProb: 0.5, xiuProb: 0.5 };
  return { taiProb: trans[cur].Tài / tot, xiuProb: trans[cur].Xỉu / tot };
}

function smartBridgeBreak(hist) {
  if (hist.length < 4) return { pred: 0, prob: 0, reason: "Thiếu dữ liệu" };
  const { streak, current, breakProb } = detectStreakAndBreak(hist);
  const last25 = hist.slice(-25).map(x => x.result);
  const scores = hist.slice(-25).map(x => x.total || 0);
  const avg = scores.reduce((s, v) => s + v, 0) / (scores.length || 1);
  const varc = scores.reduce((s, v) => s + (v - avg) ** 2, 0) / (scores.length || 1);

  // n-gram 4 trên kết quả Tài/Xỉu
  const pc = {};
  for (let i = 0; i <= last25.length - 4; i++) {
    const k = last25.slice(i, i + 4).join(",");
    pc[k] = (pc[k] || 0) + 1;
  }
  const common = Object.entries(pc).sort((a, b) => b[1] - a[1])[0];
  const hasRep = common && common[1] >= 4;

  let finalProb = breakProb;
  let reason = "";

  if (streak >= 7) {
    finalProb = Math.min(finalProb + 0.2, 0.95);
    reason = `[Bẻ cầu] Chuỗi ${streak} ${current} dài`;
  } else if (streak >= 5 && varc > 9) {
    finalProb = Math.min(finalProb + 0.15, 0.9);
    reason = `[Bẻ cầu] Biến thiên điểm cao (${r01(varc)})`;
  } else if (hasRep && last25.slice(-6).every(r => r === current)) {
    finalProb = Math.min(finalProb + 0.1, 0.85);
    reason = `[Bẻ cầu] Mẫu lặp mạnh`;
  } else {
    finalProb = Math.max(finalProb - 0.1, 0.2);
    reason = `[Theo cầu] Chưa có tín hiệu bẻ`;
  }

  const pred = finalProb > 0.7 ? (current === "Tài" ? "Xỉu" : "Tài") : current;
  return { pred, prob: finalProb, reason };
}

function trendAndProb(hist) {
  if (hist.length < 4) return 0;
  const { streak, current, breakProb } = detectStreakAndBreak(hist);
  if (streak >= 6) return breakProb > 0.8 ? (current === "Tài" ? "Xỉu" : "Tài") : current;
  const last20 = hist.slice(-20).map(x => x.result);
  const weights = last20.map((_, i) => Math.pow(1.3, i));
  const tW = weights.reduce((s, w, i) => s + (last20[i] === "Tài" ? w : 0), 0);
  const xW = weights.reduce((s, w, i) => s + (last20[i] === "Xỉu" ? w : 0), 0);
  if (tW + xW === 0) return last20.at(-1) === "Xỉu" ? "Tài" : "Xỉu";
  if (Math.abs(tW - xW) / (tW + xW) >= 0.3) return tW > xW ? "Tài" : "Xỉu";
  return last20.at(-1) === "Xỉu" ? "Tài" : "Xỉu";
}

function shortPattern(hist) {
  if (hist.length < 4) return 0;
  const { streak, current, breakProb } = detectStreakAndBreak(hist);
  if (streak >= 5) return breakProb > 0.8 ? (current === "Tài" ? "Xỉu" : "Tài") : current;
  const last10 = hist.slice(-10).map(x => x.result);
  // n-gram 4 ngắn
  const pc = {};
  for (let i = 0; i <= last10.length - 4; i++) {
    const k = last10.slice(i, i + 4).join(",");
    pc[k] = (pc[k] || 0) + 1;
  }
  const c = Object.entries(pc).sort((a, b) => b[1] - a[1])[0];
  if (c && c[1] >= 3) {
    const parts = c[0].split(",");
    return parts.at(-1) !== last10.at(-1) ? (last10.at(-1) === "Tài" ? "Xỉu" : "Tài") : last10.at(-1);
  }
  return last10.at(-1) === "Xỉu" ? "Tài" : "Xỉu";
}

function meanDeviation(hist) {
  if (hist.length < 4) return 0;
  const last15 = hist.slice(-15).map(x => x.result);
  const t = last15.filter(r => r === "Tài").length;
  const x = last15.length - t;
  const imb = last15.length ? Math.abs(t - x) / last15.length : 0;
  if (imb < 0.3) return last15.at(-1) === "Xỉu" ? "Tài" : "Xỉu";
  return x > t ? "Xỉu" : "Tài";
}

function recentSwitch(hist) {
  if (hist.length < 4) return 0;
  const last12 = hist.slice(-12).map(x => x.result);
  const sw = last12.slice(1).reduce((c, r, i) => c + (r !== last12[i] ? 1 : 0), 0);
  return sw >= 7 ? (last12.at(-1) === "Xỉu" ? "Tài" : "Xỉu") : (last12.at(-1) === "Xỉu" ? "Tài" : "Xỉu");
}

function aiHtddLogic(hist) {
  if (hist.length < 4) return { pred: Math.random() < 0.5 ? "Tài" : "Xỉu", reason: "Không đủ lịch sử" };

  const last7R = hist.slice(-7).map(x => x.result);
  const last4 = hist.slice(-4).map(x => x.result);
  if (last4.join(",") === "Tài,Xỉu,Tài,Xỉu") return { pred: "Tài", reason: "Mẫu 1T1X lặp" };
  if (last4.join(",") === "Xỉu,Tài,Xỉu,Tài") return { pred: "Xỉu", reason: "Mẫu 1X1T lặp" };

  const last5 = hist.slice(-5).map(x => x.result);
  if (last5.join(",") === "Tài,Tài,Xỉu,Xỉu,Tài") return { pred: "Xỉu", reason: "Mẫu 2T2X1T" };
  if (last5.join(",") === "Xỉu,Xỉu,Tài,Tài,Xỉu") return { pred: "Tài", reason: "Mẫu 2X2T1X" };

  // Streak 7
  if (hist.length >= 10 && hist.slice(-7).every(i => i.result === "Tài")) return { pred: "Xỉu", reason: "Chuỗi Tài dài" };
  if (hist.length >= 10 && hist.slice(-7).every(i => i.result === "Xỉu")) return { pred: "Tài", reason: "Chuỗi Xỉu dài" };

  // Điểm trung bình & phương sai (trên tổng)
  const scores = hist.slice(-7).map(x => x.total || 0);
  const avg = scores.reduce((s, v) => s + v, 0) / (scores.length || 1);
  const varc = scores.reduce((s, v) => s + (v - avg) ** 2, 0) / (scores.length || 1);

  if (avg > 10.5 && varc > 8) return { pred: "Xỉu", reason: `Điểm cao, variance lớn (${r01(varc)})` };
  if (avg > 10.5) return { pred: "Tài", reason: `Điểm TB cao (${r01(avg)})` };
  if (avg < 10.5 && varc > 8) return { pred: "Tài", reason: `Điểm thấp, variance lớn (${r01(varc)})` };
  if (avg < 10.5) return { pred: "Xỉu", reason: `Điểm TB thấp (${r01(avg)})` };

  // Cân bằng → random có điều kiện
  const t = last7R.filter(r => r === "Tài").length;
  const x = last7R.length - t;
  if (t > x + 2) return { pred: "Xỉu", reason: "Thiên Tài → cân bằng" };
  if (x > t + 2) return { pred: "Tài", reason: "Thiên Xỉu → cân bằng" };
  return { pred: Math.random() < 0.5 ? "Tài" : "Xỉu", reason: "Cân bằng hoàn hảo" };
}

// ====== Sử dụng Mau_cau (n-gram 'T'/'X') để dự đoán ======
function mauCauPredict() {
  if (history.length < 6) return { pred: null, weight: 0, note: "Thiếu dài chuỗi" };
  const seq = history.map(h => toTX(h.result));
  const window = seq.slice(-5); // lấy 5 ký tự gần nhất để thử n=5,4,3
  const tries = [
    { n: 5, key: window.slice(-5).join("") },
    { n: 4, key: window.slice(-4).join("") },
    { n: 3, key: window.slice(-3).join("") },
  ];

  for (const t of tries) {
    const mem = patternMemory[t.n]?.[t.key];
    if (mem && (mem.T > 0 || mem.X > 0)) {
      const pred = mem.T === mem.X ? (Math.random() < 0.5 ? "T" : "X") : (mem.T > mem.X ? "T" : "X");
      const conf = (Math.max(mem.T, mem.X) / (mem.T + mem.X));
      return { pred: pred === "T" ? "Tài" : "Xỉu", weight: 0.28 + conf * 0.22, note: `Mẫu ${t.n}-gram (${t.key}→${pred})` };
    }
  }
  return { pred: null, weight: 0, note: "Không khớp mẫu n-gram" };
}

// ====== Hợp nhất mô-đun để ra dự đoán cuối ======
function finalPredict() {
  if (history.length === 0) {
    return { pred: Math.random() < 0.5 ? "Tài" : "Xỉu", conf: 0.5, explain: "Không có dữ liệu" };
  }

  const curSession = history.at(-1).session;

  // Lấy dự đoán từng mô-đun
  const tPred = trendAndProb(history);             // "Tài"/"Xỉu"
  const sPred = shortPattern(history);
  const mPred = meanDeviation(history);
  const swPred = recentSwitch(history);
  const br = smartBridgeBreak(history);
  const mk = markovTransitionProb(history);
  const ai = aiHtddLogic(history);
  const mc = mauCauPredict();

  // Gán weight (đã tinh chỉnh)
  const perf = (name, lb = 15) => evaluatePerformance(name, lb);
  const weights = {
    trend: 0.18 * perf("trend"),
    short: 0.18 * perf("short"),
    mean:  0.22 * perf("mean"),
    switch:0.18 * perf("switch"),
    bridge:0.14 * perf("bridge"),
    markov:0.10 * perf("markov"),
    aihtdd:0.20 * perf("aihtdd"),
    mau:   mc.weight || 0.22, // weight linh hoạt theo độ mạnh pattern
  };

  let taiScore = 0, xiuScore = 0;
  const add = (p, w) => { if (p === "Tài") taiScore += w; else if (p === "Xỉu") xiuScore += w; };

  add(tPred, weights.trend);
  add(sPred, weights.short);
  add(mPred, weights.mean);
  add(swPred, weights.switch);
  add(br.pred, weights.bridge);
  add(mk.taiProb > mk.xiuProb ? "Tài" : "Xỉu", weights.markov);
  add(ai.pred, weights.aihtdd);
  if (mc.pred) add(mc.pred, weights.mau);

  // Momentum: ưu tiên kết quả gần đây
  const last5 = history.slice(-5).map(x => x.result);
  const mom = (last5.filter(r => r === "Tài").length > 2) ? 0.15 : -0.15;
  if (mom > 0) taiScore += mom; else xiuScore += Math.abs(mom);

  // Nếu mẫu xấu (nhiễu cao) thì giảm đều
  if (isBadPattern(history)) { taiScore *= 0.75; xiuScore *= 0.75; }

  // Điều chỉnh thiên lệch 15 phiên gần
  const last15 = history.slice(-15).map(x => x.result);
  const t15 = last15.filter(r => r === "Tài").length;
  if (t15 >= 10) xiuScore += 0.2; else if (t15 <= 5) taiScore += 0.2;

  const pred = taiScore >= xiuScore ? "Tài" : "Xỉu";
  const conf = Math.min(0.98, Math.max(0.52, 0.5 + Math.abs(taiScore - xiuScore))); // 0.52–0.98

  // Lưu lại để chấm hiệu năng sau này
  if (!modelPredictions[curSession]) modelPredictions[curSession] = {};
  modelPredictions[curSession] = {
    trend: tPred, short: sPred, mean: mPred, switch: swPred, bridge: br.pred,
    markov: mk.taiProb > mk.xiuProb ? "Tài" : "Xỉu",
    aihtdd: ai.pred, mau: mc.pred || "None",
    final: pred
  };

  const explain = [
    `AI: ${ai.reason}`,
    `Bridge: ${br.reason} (p=${r01(br.prob)})`,
    `Markov T=${r01(mk.taiProb)} X=${r01(mk.xiuProb)}`,
    `Mẫu_cầu: ${mc.note}`,
    `Điểm T=${r01(taiScore)} · X=${r01(xiuScore)}`
  ].join(" | ");

  return { pred, conf: r01(conf), explain };
}

function isBadPattern(hist) {
  const last20 = hist.slice(-20).map(x => x.result);
  const sw = last20.slice(1).reduce((c, r, i) => c + (r !== last20[i] ? 1 : 0), 0);
  const { streak } = detectStreakAndBreak(hist);
  return sw >= 12 || streak >= 11;
}

// Chấm “hiệu năng mô-đun” (nhẹ) để tự điều chỉnh weight
function evaluatePerformance(modelName, lookback = 15) {
  const keys = Object.keys(modelPredictions).map(k => Number(k)).sort((a, b) => a - b);
  if (keys.length < 2) return 1;
  const use = keys.slice(-lookback - 1); // cần phiên-1 để so phiên hiện thực
  let correct = 0, total = 0;

  for (let i = 1; i < use.length; i++) {
    const sessPrev = use[i - 1];     // dự đoán cho sessPrev -> so với kết quả ở sessPrev (đã biết)
    const sessNow = use[i];          // bước qua phiên mới
    const m = modelPredictions[sessPrev];
    const actual = history.find(h => h.session === sessPrev)?.result;
    if (!m || !actual) continue;
    const pred = m[modelName];
    if (pred === actual) correct++;
    total++;
  }
  if (!total) return 1;
  const ratio = 1 + (correct - total / 2) / (total / 2);
  return Math.max(0.6, Math.min(1.6, ratio));
}

// ====== Endpoint chính ======
app.get("/api/custom", async (req, res) => {
  try {
    await loadSource(); // cập nhật history & patternMemory

    if (history.length === 0) {
      return res.status(500).json({ error: "Không có dữ liệu lịch sử" });
    }

    const latest = history.at(-1);
    const { session, dice, total, result } = latest;

    // Mau_cau: 1 phiên lưu 1 ký tự theo kết quả phiên đó
    const Mau_cau = result === "Tài" ? "T" : "X";

    // Dự đoán cho PHIÊN SAU
    const { pred, conf, explain } = finalPredict();

    res.json({
      id: "Tele@idol_vannhat",
      Phien_truoc: session,
      Phien_sau: session + 1,
      Xuc_xac: dice,
      Tong: total,
      Ket_qua: result,
      Du_doan: pred,
      Do_tin_cay: conf,
      Giai_thich: explain,
      Mau_cau: Mau_cau
    });
  } catch (err) {
    res.status(500).json({ error: "Lỗi xử lý", detail: err.message });
  }
});

// ====== Tiện ích: xem nhanh log/health ======
app.get("/", (_req, res) => res.send("Taixiu SIÊU VIP API ok. Use /api/custom"));
app.get("/debug/history", (_req, res) => res.json(history.slice(-50)));
app.get("/debug/pattern", (_req, res) => res.json(patternMemory));

app.listen(PORT, () => {
  console.log(`✅ API chạy: http://localhost:${PORT}/api/custom`);
});