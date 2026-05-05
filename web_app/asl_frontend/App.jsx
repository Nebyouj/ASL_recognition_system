import { useState, useEffect, useRef, useCallback } from "react";

// ─── Constants ───────────────────────────────────────────────────────────────
const WS_URL = "ws://localhost:8000/ws";
const API_URL = "http://localhost:8000";
const FRAME_INTERVAL = 80;

const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],
  [0,17],
];

const LANGUAGES = [
  { code: "en", label: "English" },
  { code: "fr", label: "French" },
  { code: "ar", label: "Arabic" },
  { code: "de", label: "German" },
];

// ─── Upper body only ──────────────────────────────────────────────────────────
function drawBody(ctx, w, h) {
  const cx       = w / 2;
  const headY    = h * 0.12;
  const shoulderY = h * 0.30;
  const waistY   = h * 0.62;
  const shoulderW = w * 0.20;

  ctx.strokeStyle = "rgba(160,200,255,0.30)";
  ctx.lineWidth = 3;
  ctx.lineCap = "round";

  // Head
  ctx.beginPath();
  ctx.arc(cx, headY, w * 0.06, 0, Math.PI * 2);
  ctx.stroke();

  // Neck → shoulder line
  ctx.beginPath();
  ctx.moveTo(cx, headY + w * 0.06);
  ctx.lineTo(cx, shoulderY);
  ctx.stroke();

  // Shoulders
  ctx.beginPath();
  ctx.moveTo(cx - shoulderW, shoulderY);
  ctx.lineTo(cx + shoulderW, shoulderY);
  ctx.stroke();

  // Torso to waist
  ctx.beginPath();
  ctx.moveTo(cx, shoulderY);
  ctx.lineTo(cx, waistY);
  ctx.stroke();

  // Upper arms (forearms replaced by animated hands)
  ctx.beginPath();
  ctx.moveTo(cx - shoulderW, shoulderY);
  ctx.lineTo(cx - shoulderW * 1.6, shoulderY + h * 0.18);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(cx + shoulderW, shoulderY);
  ctx.lineTo(cx + shoulderW * 1.6, shoulderY + h * 0.18);
  ctx.stroke();

  // Waist line
  ctx.beginPath();
  ctx.moveTo(cx - shoulderW * 0.7, waistY);
  ctx.lineTo(cx + shoulderW * 0.7, waistY);
  ctx.stroke();
}

function drawSkeleton(ctx, points, color) {
  ctx.save();
  ctx.shadowColor = color;
  ctx.shadowBlur = 12;
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.lineCap = "round";

  for (const [a, b] of HAND_CONNECTIONS) {
    const p1 = points[a], p2 = points[b];
    if (!p1 || !p2) continue;
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();
  }

  ctx.fillStyle = "#fff";
  for (const p of points) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

// Hands placed at waist level, offset left/right
function scaleHandPoints(rawPoints, canvasW, canvasH, offsetX = 0) {
  const xs = rawPoints.map(p => p.x);
  const ys = rawPoints.map(p => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const bw = maxX - minX, bh = maxY - minY;
  if (bw === 0 || bh === 0) return rawPoints;

  const scale = Math.min(canvasW * 0.25 / bw, canvasH * 0.30 / bh);
  const targetX = canvasW / 2 + offsetX;
  const targetY = canvasH * 0.58; // waist level — where upper arms end

  return rawPoints.map(p => ({
    x: (p.x - minX - bw / 2) * scale + targetX,
    y: (p.y - minY - bh / 2) * scale + targetY,
  }));
}

function parseHand(flat42) {
  const pts = [];
  for (let i = 0; i < flat42.length; i += 2)
    pts.push({ x: flat42[i], y: flat42[i + 1] });
  return pts;
}

function interpolate(a, b, t) {
  return a.map((v, i) => v + (b[i] - v) * t);
}

function buildAllFrames(sequences) {
  const all = [];
  for (const seq of sequences) {
    for (let i = 0; i < seq.length - 1; i++) {
      all.push(seq[i]);
      all.push(interpolate(seq[i], seq[i + 1], 0.33));
      all.push(interpolate(seq[i], seq[i + 1], 0.66));
    }
    if (seq.length > 0) all.push(seq[seq.length - 1]);
  }
  return all;
}

// ─── Sub-components ───────────────────────────────────────────────────────────
function StatusDot({ connected }) {
  return (
    <span style={{
      display: "inline-block",
      width: 10, height: 10,
      borderRadius: "50%",
      background: connected ? "#22c55e" : "#ef4444",
      boxShadow: connected ? "0 0 8px #22c55e" : "0 0 8px #ef4444",
      marginRight: 8,
      flexShrink: 0,
    }} />
  );
}

function WordChip({ word, index }) {
  return (
    <span style={{
      display: "inline-block",
      background: "rgba(99,179,237,0.15)",
      border: "1px solid rgba(99,179,237,0.4)",
      borderRadius: 8,
      padding: "4px 10px",
      margin: "3px",
      fontSize: 18,
      fontFamily: "'DM Mono', monospace",
      color: "#e2e8f0",
      animation: "chipIn 0.3s ease",
      animationFillMode: "both",
      animationDelay: `${index * 0.04}s`,
    }}>
      {word}
    </span>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const videoRef      = useRef(null);
  const avatarRef     = useRef(null);
  const wsRef         = useRef(null);
  const frameTimerRef = useRef(null);
  const animFrameRef  = useRef(null);
  const lastWordRef   = useRef("");        // ref so WS closure always sees latest
  const outputLangRef = useRef("en");     // ref so WS closure always sees latest lang

  const [connected,    setConnected]    = useState(false);
  const [sentence,     setSentence]     = useState([]);
  const [translated,   setTranslated]   = useState("");
  const [outputLang,   setOutputLang]   = useState("en");

  // Reverse ASL
  const [reverseInput,  setReverseInput]  = useState("");
  const [reverseLang,   setReverseLang]   = useState("en");
  const [reverseStatus, setReverseStatus] = useState("");
  const [isAnimating,   setIsAnimating]   = useState(false);

  // Keep ref in sync with state so the WS closure doesn't go stale
  useEffect(() => { outputLangRef.current = outputLang; }, [outputLang]);

  // ── Translation ─────────────────────────────────────────────────────────
  const translateSentence = useCallback(async (text, lang) => {
    if (!text || lang === "en") { setTranslated(""); return; }
    try {
      const res = await fetch(`${API_URL}/translate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, lang }),
      });
      const t = await res.json();
      setTranslated(t.translated || "");
    } catch (_) {}
  }, []);

  // ── Frame loop ───────────────────────────────────────────────────────────
  const stopFrameLoop = useCallback(() => {
    if (frameTimerRef.current) clearInterval(frameTimerRef.current);
  }, []);

  const startFrameLoop = useCallback((ws) => {
    stopFrameLoop();
    frameTimerRef.current = setInterval(() => {
      const video = videoRef.current;
      if (!video || video.readyState < 2 || ws.readyState !== WebSocket.OPEN) return;

      const canvas = document.createElement("canvas");
      canvas.width  = 320;
      canvas.height = 240;
      canvas.getContext("2d").drawImage(video, 0, 0, 320, 240);

      canvas.toBlob((blob) => {
        const reader = new FileReader();
        reader.onload = () => {
          const b64 = reader.result.split(",")[1];
          if (ws.readyState === WebSocket.OPEN) ws.send(b64);
        };
        reader.readAsDataURL(blob);
      }, "image/jpeg", 0.7);
    }, FRAME_INTERVAL);
  }, [stopFrameLoop]);

  // ── WebSocket ────────────────────────────────────────────────────────────
  const connectWS = useCallback(() => {
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      startFrameLoop(ws);
    };

    ws.onmessage = (e) => {
      const data = JSON.parse(e.data);

      if (data.sentence !== undefined) {
        const words = data.sentence.trim().split(" ").filter(Boolean);

        setSentence(prev => {
          const prevStr = prev.join(" ");
          const newStr  = words.join(" ");

          // Only translate when sentence actually gains a new word
          if (prevStr !== newStr && newStr.length > 0) {
            const lang = outputLangRef.current;
            if (lang !== "en") translateSentence(newStr, lang);
            else setTranslated("");
          }

          return words;
        });

        // TTS — only speak new words
        if (data.word && data.word !== lastWordRef.current) {
          lastWordRef.current = data.word;
          playTTS(data.word);
        }
      }
    };

    ws.onclose = () => {
      setConnected(false);
      stopFrameLoop();
      setTimeout(connectWS, 2000);
    };

    ws.onerror = () => ws.close();
  }, [startFrameLoop, stopFrameLoop, translateSentence]);

  // ── Camera + WS init ─────────────────────────────────────────────────────
  useEffect(() => {
    navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
      if (videoRef.current) videoRef.current.srcObject = stream;
    });
    connectWS();
    return () => {
      stopFrameLoop();
      wsRef.current?.close();
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── TTS ──────────────────────────────────────────────────────────────────
  const playTTS = (word) => {
    const audio = document.getElementById("tts-audio");
    if (!audio) return;
    audio.src = `${API_URL}/tts?sentence=${encodeURIComponent(word)}`;
    audio.play().catch(() => {});
  };

  // ── Clear ────────────────────────────────────────────────────────────────
  const clearSentence = async () => {
    setSentence([]);
    setTranslated("");
    lastWordRef.current = "";
    try { await fetch(`${API_URL}/clear`, { method: "POST" }); } catch (_) {}
  };

  // ── Reverse ASL ──────────────────────────────────────────────────────────
  const playReverseASL = async () => {
    if (!reverseInput.trim() || isAnimating) return;
    setIsAnimating(true);

    let textToSign = reverseInput.trim();

    if (reverseLang !== "en") {
      setReverseStatus("Translating to English...");
      try {
        const res = await fetch(`${API_URL}/translate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: textToSign, lang: "en", source_lang: reverseLang }),
        });
        const t = await res.json();
        textToSign = t.translated || textToSign;
      } catch (_) {}
    }

    setReverseStatus(`Signing: "${textToSign}"`);

    try {
      const res  = await fetch(`${API_URL}/reverse_landmarks?sentence=${encodeURIComponent(textToSign)}`);
      const data = await res.json();

      if (!data.sequences || data.sequences.length === 0) {
        setReverseStatus("⚠ No data found for those words");
        setIsAnimating(false);
        return;
      }

      animateFrames(buildAllFrames(data.sequences), () => {
        setReverseStatus("✓ Done");
        setIsAnimating(false);
      });
    } catch (_) {
      setReverseStatus("Error fetching landmarks");
      setIsAnimating(false);
    }
  };

  const animateFrames = (frames, onDone) => {
    const canvas = avatarRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width, h = canvas.height;
    let i = 0;

    if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);

    const draw = () => {
      if (i >= frames.length) { onDone(); return; }

      ctx.clearRect(0, 0, w, h);

      // Subtle grid
      ctx.strokeStyle = "rgba(99,179,237,0.05)";
      ctx.lineWidth = 1;
      for (let gx = 0; gx < w; gx += 40) {
        ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, h); ctx.stroke();
      }
      for (let gy = 0; gy < h; gy += 40) {
        ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke();
      }

      drawBody(ctx, w, h);

      const frame = frames[i];
      const leftRaw  = parseHand(frame.slice(0, 42));
      const rightRaw = parseHand(frame.slice(42));

      if (leftRaw.some(p => p.x !== 0 || p.y !== 0))
        drawSkeleton(ctx, scaleHandPoints(leftRaw,  w, h, -w * 0.18), "#63b3ed");

      if (rightRaw.some(p => p.x !== 0 || p.y !== 0))
        drawSkeleton(ctx, scaleHandPoints(rightRaw, w, h, +w * 0.18), "#f6ad55");

      i++;
      animFrameRef.current = requestAnimationFrame(draw);
    };

    draw();
  };

  // ─── Render ───────────────────────────────────────────────────────────────
  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;800&display=swap');

        @keyframes chipIn {
          from { opacity: 0; transform: translateY(6px) scale(0.92); }
          to   { opacity: 1; transform: translateY(0) scale(1); }
        }
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50%       { opacity: 0.5; }
        }

        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        body {
          background: #050d1a;
          color: #e2e8f0;
          font-family: 'Syne', sans-serif;
          min-height: 100vh;
        }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(99,179,237,0.3); border-radius: 3px; }
      `}</style>

      <audio id="tts-audio" style={{ display: "none" }} />

      <div style={{
        minHeight: "100vh",
        background: "radial-gradient(ellipse at 20% 20%, rgba(30,60,120,0.6) 0%, transparent 60%), radial-gradient(ellipse at 80% 80%, rgba(10,30,80,0.8) 0%, transparent 60%), #050d1a",
        padding: "24px",
        display: "flex",
        flexDirection: "column",
        gap: 24,
      }}>

        {/* ── Header ── */}
        <div style={{ textAlign: "center", paddingBottom: 8 }}>
          <h1 style={{
            fontFamily: "'Syne', sans-serif",
            fontWeight: 800,
            fontSize: "clamp(22px, 4vw, 38px)",
            letterSpacing: "-0.02em",
            background: "linear-gradient(135deg, #63b3ed 0%, #a78bfa 50%, #f6ad55 100%)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
          }}>
            ASL Bidirectional Communication
          </h1>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 6, marginTop: 8, opacity: 0.7, fontSize: 13 }}>
            <StatusDot connected={connected} />
            {connected ? "WebSocket connected — real-time" : "Reconnecting..."}
          </div>
        </div>

        {/* ── Grid ── */}
        <div style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(480px, 1fr))",
          gap: 20,
        }}>

          {/* ── LEFT: Recognition ── */}
          <div style={panelStyle}>
            <SectionLabel>Live ASL → Text</SectionLabel>

            <div style={{ position: "relative", borderRadius: 12, overflow: "hidden", background: "#000" }}>
              <video
                ref={videoRef}
                autoPlay playsInline muted
                style={{ width: "100%", display: "block", borderRadius: 12 }}
              />
              {!connected && (
                <div style={{
                  position: "absolute", inset: 0,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  background: "rgba(0,0,0,0.6)",
                  fontSize: 14, color: "#f6ad55",
                  animation: "pulse 1.5s infinite",
                }}>
                  Connecting to server...
                </div>
              )}
            </div>

            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <span style={{ fontSize: 13, opacity: 0.6, flexShrink: 0 }}>Translate to:</span>
              <select
                value={outputLang}
                onChange={e => setOutputLang(e.target.value)}
                style={selectStyle}
              >
                {LANGUAGES.map(l => (
                  <option key={l.code} value={l.code}>{l.label}</option>
                ))}
              </select>
            </div>

            <div style={boxStyle}>
              <div style={labelStyle}>Recognized sentence</div>
              <div style={{ minHeight: 40, lineHeight: 1.8 }}>
                {sentence.length === 0
                  ? <span style={{ opacity: 0.3, fontFamily: "'DM Mono', monospace", fontSize: 14 }}>Start signing...</span>
                  : sentence.map((w, i) => <WordChip key={i} word={w} index={i} />)
                }
              </div>
            </div>

            {/* Only show translation box when language is non-English AND there's a sentence */}
            {outputLang !== "en" && sentence.length > 0 && (
              <div style={boxStyle}>
                <div style={labelStyle}>
                  Translation ({LANGUAGES.find(l => l.code === outputLang)?.label})
                </div>
                <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 16, color: "#a78bfa", minHeight: 28 }}>
                  {translated
                    ? translated
                    : <span style={{ opacity: 0.4, fontSize: 13 }}>Translating...</span>
                  }
                </div>
              </div>
            )}

            <button onClick={clearSentence} style={btnStyle("#ef4444")}>
              Clear
            </button>
          </div>

          {/* ── RIGHT: Reverse ASL ── */}
          <div style={panelStyle}>
            <SectionLabel>Text → ASL Avatar</SectionLabel>

            <div style={{ display: "flex", gap: 8 }}>
              <select
                value={reverseLang}
                onChange={e => setReverseLang(e.target.value)}
                style={{ ...selectStyle, width: "auto", flexShrink: 0 }}
              >
                {LANGUAGES.map(l => (
                  <option key={l.code} value={l.code}>{l.label}</option>
                ))}
              </select>
              <input
                value={reverseInput}
                onChange={e => setReverseInput(e.target.value)}
                onKeyDown={e => e.key === "Enter" && playReverseASL()}
                placeholder="Type a word or sentence..."
                style={inputStyle}
              />
            </div>

            {reverseLang !== "en" && (
              <div style={{ fontSize: 12, opacity: 0.5, paddingLeft: 4 }}>
                Will translate to English first, then sign
              </div>
            )}

            <button
              onClick={playReverseASL}
              disabled={isAnimating || !reverseInput.trim()}
              style={btnStyle(isAnimating ? "#475569" : "#63b3ed", isAnimating)}
            >
              {isAnimating ? "Animating..." : "Animate Sign ▶"}
            </button>

            {reverseStatus && (
              <div style={{ fontSize: 13, color: "#f6ad55", opacity: 0.8, paddingLeft: 4 }}>
                {reverseStatus}
              </div>
            )}

            <div style={{
              position: "relative",
              borderRadius: 12,
              overflow: "hidden",
              background: "#020a15",
              border: "1px solid rgba(99,179,237,0.15)",
            }}>
              <canvas
                ref={avatarRef}
                width={640}
                height={420}
                style={{ width: "100%", display: "block" }}
              />
              {!isAnimating && (
                <div style={{
                  position: "absolute", inset: 0,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  pointerEvents: "none",
                  opacity: 0.15, fontSize: 13,
                  fontFamily: "'DM Mono', monospace",
                }}>
                  Avatar will appear here
                </div>
              )}
            </div>

            <div style={{ display: "flex", gap: 16, fontSize: 12, opacity: 0.5 }}>
              <span><span style={{ color: "#63b3ed" }}>●</span> Left hand</span>
              <span><span style={{ color: "#f6ad55" }}>●</span> Right hand</span>
              <span><span style={{ color: "rgba(160,200,255,0.5)" }}>—</span> Body</span>
            </div>
          </div>

        </div>
      </div>
    </>
  );
}

// ─── Style constants ──────────────────────────────────────────────────────────
const panelStyle = {
  background: "rgba(255,255,255,0.03)",
  border: "1px solid rgba(99,179,237,0.1)",
  borderRadius: 16,
  padding: 20,
  display: "flex",
  flexDirection: "column",
  gap: 12,
  backdropFilter: "blur(8px)",
};

const boxStyle = {
  background: "rgba(0,0,0,0.3)",
  border: "1px solid rgba(99,179,237,0.1)",
  borderRadius: 10,
  padding: "12px 14px",
};

const labelStyle = {
  fontSize: 11,
  letterSpacing: "0.1em",
  textTransform: "uppercase",
  opacity: 0.45,
  marginBottom: 8,
  fontFamily: "'DM Mono', monospace",
};

const selectStyle = {
  width: "100%",
  background: "rgba(0,0,0,0.4)",
  border: "1px solid rgba(99,179,237,0.2)",
  borderRadius: 8,
  color: "#e2e8f0",
  padding: "8px 12px",
  fontSize: 14,
  fontFamily: "'Syne', sans-serif",
  cursor: "pointer",
  outline: "none",
};

const inputStyle = {
  flex: 1,
  background: "rgba(0,0,0,0.4)",
  border: "1px solid rgba(99,179,237,0.2)",
  borderRadius: 8,
  color: "#e2e8f0",
  padding: "8px 12px",
  fontSize: 14,
  fontFamily: "'Syne', sans-serif",
  outline: "none",
};

const btnStyle = (color, disabled = false) => ({
  background: disabled ? "#1e293b" : `${color}22`,
  border: `1px solid ${disabled ? "#334155" : color}`,
  borderRadius: 8,
  color: disabled ? "#475569" : color,
  padding: "10px 16px",
  fontSize: 14,
  fontFamily: "'Syne', sans-serif",
  fontWeight: 600,
  cursor: disabled ? "not-allowed" : "pointer",
  transition: "all 0.2s",
  letterSpacing: "0.02em",
});

function SectionLabel({ children }) {
  return (
    <div style={{
      fontWeight: 600,
      fontSize: 15,
      letterSpacing: "0.04em",
      color: "#94a3b8",
      paddingBottom: 4,
      borderBottom: "1px solid rgba(99,179,237,0.1)",
    }}>
      {children}
    </div>
  );
}
