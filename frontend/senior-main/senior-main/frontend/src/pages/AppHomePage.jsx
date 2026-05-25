import { useEffect, useRef, useState } from "react";
import { io } from "socket.io-client";
import { useAuth } from "../context/AuthContext";
import api from "../services/api";

function AppHomePage() {
  const { user, logout } = useAuth();
  const [activeTab, setActiveTab] = useState("chat");
  const [language, setLanguage] = useState("en");
  const [mode, setMode] = useState("letters");
  const [predictedText, setPredictedText] = useState("");
  const [status, setStatus] = useState("Camera is off");
  const [sending, setSending] = useState(false);
  const [cameraOn, setCameraOn] = useState(false);
  const [messages, setMessages] = useState([]);
  const [users, setUsers] = useState([]);
  const [selectedUserId, setSelectedUserId] = useState("");
  const [roomId, setRoomId] = useState("");
  const [chatStatus, setChatStatus] = useState("Choose a user to start chat.");

  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const socketRef = useRef(null);

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraOn(false);
    setStatus("Camera is off");
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setCameraOn(true);
      setStatus("Camera is ready");
    } catch (_error) {
      setStatus("Cannot access camera. Please allow webcam permission.");
    }
  };

  const captureImageBase64 = () => {
    const video = videoRef.current;
    if (!video || !cameraOn) return null;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.85);
  };

  const handlePredict = async () => {
    const imageBase64 = captureImageBase64();
    if (!imageBase64) {
      setStatus("Start camera first, then click Translate.");
      return;
    }

    setSending(true);
    setStatus("Translating sign...");
    try {
      const res = await api.post("/predict", { language, mode, imageBase64 });
      const text = res.data?.prediction?.text || "";
      setPredictedText((prev) => (prev ? `${prev} ${text}` : text));
      setStatus(
        `Sign translated: ${text} (${Math.round((res.data?.prediction?.confidence || 0) * 100)}%)`
      );
    } catch (err) {
      setStatus(err.response?.data?.message || "Translation failed");
    } finally {
      setSending(false);
    }
  };

  const handleSendToChat = () => {
    if (!predictedText.trim()) return;
    if (!roomId || !socketRef.current) {
      setChatStatus("Join a chat first.");
      return;
    }

    socketRef.current.emit(
      "chat:send",
      { roomId, content: predictedText, contentType: "TEXT" },
      (ack) => {
        if (!ack?.ok) {
          setChatStatus(ack?.message || "Failed to send message");
          return;
        }
        setPredictedText("");
      }
    );
  };

  const joinChat = () => {
    if (!selectedUserId || !socketRef.current) {
      setChatStatus("Select a user first.");
      return;
    }

    socketRef.current.emit("chat:join", { targetUserId: selectedUserId }, (ack) => {
      if (!ack?.ok) {
        setChatStatus(ack?.message || "Failed to join room.");
        return;
      }

      const chosen = users.find((item) => item.id === selectedUserId);
      setRoomId(ack.roomId);
      setMessages(ack.messages || []);
      setChatStatus(`Chat joined with ${chosen?.name || "user"}.`);
    });
  };

  useEffect(() => stopCamera, []);

  useEffect(() => {
    api
      .get("/users")
      .then((res) => {
        const list = res.data?.users || [];
        setUsers(list);
        if (list.length > 0) {
          setSelectedUserId(list[0].id);
        }
      })
      .catch(() => setChatStatus("Could not load users."));
  }, []);

  useEffect(() => {
    const token = localStorage.getItem("slr_token");
    if (!token) return undefined;

    const socket = io("http://localhost:5000", { auth: { token } });

    socket.on("connect", () => {
      setChatStatus("Connected. Select a user to chat.");
    });

    socket.on("chat:receive", (message) => {
      setMessages((prev) => {
        if (prev.some((item) => item.id === message.id)) return prev;
        return [...prev, message];
      });
    });

    socket.on("connect_error", (error) => {
      setChatStatus(`Socket error: ${error.message}`);
    });

    socketRef.current = socket;
    return () => socket.disconnect();
  }, []);

  return (
    <main className="page-shell">
      <div className="card wide">
        <div className="header-row">
          <h2>Hello, {user?.name}</h2>
          <button type="button" onClick={logout}>
            Logout
          </button>
        </div>

        <div className="tabs">
          <button
            className={activeTab === "chat" ? "active" : ""}
            onClick={() => setActiveTab("chat")}
            type="button"
          >
            Chatting
          </button>
          <button
            className={activeTab === "educational" ? "active" : ""}
            onClick={() => setActiveTab("educational")}
            type="button"
          >
            Educational
          </button>
        </div>

        {activeTab === "chat" ? (
          <div style={{ display: "grid", gridTemplateColumns: "1.5fr 1fr", gap: "24px" }}>
            <section className="tab-panel">
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
                <div>
                  <label className="label">Language</label>
                  <select value={language} onChange={(e) => setLanguage(e.target.value)}>
                    <option value="en">English (ASL)</option>
                    <option value="ar">Arabic (ArSL)</option>
                  </select>
                </div>
                <div>
                  <label className="label">Mode</label>
                  <select value={mode} onChange={(e) => setMode(e.target.value)}>
                    <option value="letters">Letters</option>
                    <option value="words">Words</option>
                  </select>
                </div>
              </div>

              <div className="video-placeholder" style={{ position: "relative", width: "100%", aspectRatio: "4/3" }}>
                <video ref={videoRef} autoPlay playsInline muted className="camera-view" />
                {!cameraOn && (
                  <div
                    style={{
                      position: "absolute",
                      top: "50%",
                      left: "50%",
                      transform: "translate(-50%, -50%)",
                      zIndex: 10,
                    }}
                  >
                    Camera Off
                  </div>
                )}
              </div>

              <div className="controls">
                <button type="button" onClick={startCamera}>
                  Start Camera
                </button>
                <button type="button" onClick={stopCamera}>
                  End Camera
                </button>
                <button type="button" onClick={handlePredict} disabled={sending}>
                  {sending ? "Translating..." : "Translate Sign"}
                </button>
              </div>

              <p className="status-text">{status}</p>
            </section>

            <section
              className="chat-section"
              style={{
                display: "flex",
                flexDirection: "column",
                background: "rgba(15, 23, 42, 0.4)",
                borderRadius: "16px",
                padding: "16px",
              }}
            >
              <h3 style={{ margin: "0 0 16px 0", fontSize: "1.1rem", color: "#A78BFA" }}>Live Chat</h3>

              <div style={{ display: "flex", gap: "8px", marginBottom: "12px" }}>
                <select value={selectedUserId} onChange={(e) => setSelectedUserId(e.target.value)} style={{ flex: 1 }}>
                  {users.length === 0 && <option value="">No users available</option>}
                  {users.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.name} ({item.email})
                    </option>
                  ))}
                </select>
                <button type="button" onClick={joinChat} style={{ width: "auto" }}>
                  Join
                </button>
              </div>

              <div
                className="chat-log"
                style={{
                  flex: 1,
                  overflowY: "auto",
                  minHeight: "300px",
                  maxHeight: "400px",
                  display: "flex",
                  flexDirection: "column",
                  gap: "12px",
                  paddingRight: "8px",
                  marginBottom: "16px",
                }}
              >
                {messages.length === 0 ? (
                  <p
                    style={{
                      color: "rgba(255,255,255,0.4)",
                      textAlign: "center",
                      marginTop: "auto",
                      marginBottom: "auto",
                    }}
                  >
                    No messages yet. Join a user and start chatting.
                  </p>
                ) : (
                  messages.map((msg) => {
                    const isMe = msg.sender?.id === user.id;
                    return (
                      <div
                        key={msg.id}
                        style={{ alignSelf: isMe ? "flex-end" : "flex-start", maxWidth: "85%" }}
                      >
                        <div
                          style={{
                            fontSize: "0.75rem",
                            color: "#94A3B8",
                            marginBottom: "4px",
                            textAlign: isMe ? "right" : "left",
                          }}
                        >
                          {msg.sender?.name || "Unknown"}
                        </div>
                        <div
                          style={{
                            background: isMe
                              ? "linear-gradient(135deg, #4F46E5, #818CF8)"
                              : "rgba(255, 255, 255, 0.1)",
                            padding: "10px 14px",
                            borderRadius: "12px",
                            borderBottomRightRadius: isMe ? "4px" : "12px",
                            borderTopLeftRadius: !isMe ? "4px" : "12px",
                          }}
                        >
                          {msg.content}
                        </div>
                      </div>
                    );
                  })
                )}
              </div>

              <div style={{ display: "flex", gap: "8px", marginTop: "auto" }}>
                <input
                  value={predictedText}
                  onChange={(e) => setPredictedText(e.target.value)}
                  placeholder="Type or translate a sign..."
                  style={{ flex: 1 }}
                />
                <button
                  type="button"
                  onClick={handleSendToChat}
                  disabled={!predictedText.trim()}
                  style={{ width: "auto", padding: "10px 16px", background: "#EC4899" }}
                >
                  Send
                </button>
              </div>
              <p className="status-text" style={{ marginTop: "8px" }}>
                {chatStatus}
              </p>
            </section>
          </div>
        ) : (
          <section className="tab-panel">
            <h3>Educational module</h3>
            <p>Exercises and guided learning will be added in the next steps.</p>
          </section>
        )}
      </div>
    </main>
  );
}

export default AppHomePage;
