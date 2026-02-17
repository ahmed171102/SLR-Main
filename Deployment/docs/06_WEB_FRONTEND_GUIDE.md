# 06 — Web Frontend: Step-by-Step Build Guide

> Build a React + TypeScript web app with camera, MediaPipe hand detection, and real-time sign language recognition.

---

## Overview

The web app runs in the browser and:
1. Opens the user's webcam
2. Uses **MediaPipe Hands JS** (runs in browser, no server needed for detection)
3. Extracts 21 hand landmarks → 63 floats
4. Sends landmarks to backend via WebSocket for prediction
5. Displays: predicted letter/word, confidence, built sentence
6. Supports English ↔ Arabic toggle

```
Browser Camera → MediaPipe JS → 63 floats → WebSocket → Backend → Prediction → Display
```

---

## Prerequisites

- Node.js 20 LTS installed (`node --version`)
- Backend running on `http://localhost:8000`
- VS Code with React/TypeScript extensions

---

## Step 1: Create React Project

```powershell
cd "m:\Term 10\Grad\Deployment"

# Create with Vite + React + TypeScript
npm create vite@latest web -- --template react-ts

cd web
npm install
```

---

## Step 2: Install Dependencies

```powershell
cd "m:\Term 10\Grad\Deployment\web"

# MediaPipe for hand detection
npm install @mediapipe/hands @mediapipe/camera_utils @mediapipe/drawing_utils

# UI & Routing
npm install react-router-dom

# Internationalization (English/Arabic)
npm install i18next react-i18next

# Tailwind CSS
npm install -D tailwindcss @tailwindcss/vite
```

---

## Step 3: Configure Tailwind CSS

### `tailwind.config.js`
```js
/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        primary: "#6366f1",    // Indigo
        secondary: "#10b981",  // Emerald
      }
    },
  },
  plugins: [],
}
```

### `src/index.css` — Add Tailwind directives at top:
```css
@import "tailwindcss";
```

---

## Step 4: TypeScript Types

### `src/types/index.ts`

```typescript
// Prediction types matching backend responses
export interface LetterPrediction {
  letter: string;
  confidence: number;
  above_threshold: boolean;
  top_predictions: { letter: string; confidence: number }[];
}

export interface WordPrediction {
  word: string;
  word_en: string;
  word_ar: string;
  category: string;
  confidence: number;
  above_threshold: boolean;
  top_predictions: { word_en: string; word_ar: string; confidence: number }[];
}

export interface DecoderResult {
  committed: boolean;
  text?: string;          // Letter decoder
  sentence?: string;      // Word decoder
  current_word?: string;
  event: string;
}

export interface WSResponse {
  mode: "letter" | "word";
  prediction?: LetterPrediction | WordPrediction;
  decoder?: DecoderResult;
  frames_buffered?: number;
  frames_needed?: number;
  error?: string;
}

export type Language = "en" | "ar";
```

---

## Step 5: Constants

### `src/utils/constants.ts`

```typescript
export const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
export const WS_URL = import.meta.env.VITE_WS_URL || "ws://localhost:8000/ws/recognize";

// MediaPipe settings
export const MEDIAPIPE_CONFIG = {
  maxNumHands: 1,
  modelComplexity: 1,
  minDetectionConfidence: 0.7,
  minTrackingConfidence: 0.5,
};

// Camera settings
export const CAMERA_WIDTH = 640;
export const CAMERA_HEIGHT = 480;
export const CAMERA_FPS = 30;
```

---

## Step 6: Landmark Processing Utility

### `src/utils/landmarks.ts`

```typescript
import { NormalizedLandmarkList } from "@mediapipe/hands";

/**
 * Convert MediaPipe hand landmarks to 63 normalized floats.
 * Same normalization as training: relative to wrist (landmark 0).
 */
export function extractLandmarks(handLandmarks: NormalizedLandmarkList): number[] {
  const landmarks: number[] = [];

  // Wrist position (reference point)
  const wrist = handLandmarks[0];

  for (let i = 0; i < 21; i++) {
    const lm = handLandmarks[i];
    // Relative to wrist, matching training pipeline
    landmarks.push(lm.x - wrist.x);
    landmarks.push(lm.y - wrist.y);
    landmarks.push(lm.z - wrist.z);
  }

  return landmarks; // 63 floats
}

/**
 * Check if landmarks are valid (not all zeros).
 */
export function isValidLandmarks(landmarks: number[]): boolean {
  if (landmarks.length !== 63) return false;
  const sum = landmarks.reduce((a, b) => a + Math.abs(b), 0);
  return sum > 0.01;
}
```

---

## Step 7: WebSocket Hook

### `src/hooks/useWebSocket.ts`

```typescript
import { useState, useEffect, useRef, useCallback } from "react";
import { WS_URL } from "../utils/constants";
import { WSResponse, Language } from "../types";

export function useWebSocket() {
  const [isConnected, setIsConnected] = useState(false);
  const [lastResponse, setLastResponse] = useState<WSResponse | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Connect on mount
  useEffect(() => {
    const ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      console.log("WebSocket connected");
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      const data: WSResponse = JSON.parse(event.data);
      setLastResponse(data);
    };

    ws.onclose = () => {
      console.log("WebSocket disconnected");
      setIsConnected(false);
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
    };

    wsRef.current = ws;

    return () => {
      ws.close();
    };
  }, []);

  // Send landmarks
  const sendLandmarks = useCallback((landmarks: number[], language: Language) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ landmarks, language }));
    }
  }, []);

  // Send clear command
  const sendClear = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ command: "clear" }));
    }
  }, []);

  return { isConnected, lastResponse, sendLandmarks, sendClear };
}
```

---

## Step 8: MediaPipe Hook

### `src/hooks/useMediaPipe.ts`

```typescript
import { useEffect, useRef, useState } from "react";
import { Hands, Results } from "@mediapipe/hands";
import { Camera } from "@mediapipe/camera_utils";
import { MEDIAPIPE_CONFIG, CAMERA_WIDTH, CAMERA_HEIGHT } from "../utils/constants";
import { extractLandmarks } from "../utils/landmarks";

interface UseMediaPipeProps {
  videoRef: React.RefObject<HTMLVideoElement>;
  canvasRef: React.RefObject<HTMLCanvasElement>;
  onLandmarks: (landmarks: number[]) => void;
  enabled: boolean;
}

export function useMediaPipe({ videoRef, canvasRef, onLandmarks, enabled }: UseMediaPipeProps) {
  const [isLoading, setIsLoading] = useState(true);
  const [handDetected, setHandDetected] = useState(false);
  const cameraRef = useRef<Camera | null>(null);

  useEffect(() => {
    if (!enabled || !videoRef.current || !canvasRef.current) return;

    const hands = new Hands({
      locateFile: (file) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
    });

    hands.setOptions(MEDIAPIPE_CONFIG);

    hands.onResults((results: Results) => {
      setIsLoading(false);

      // Draw on canvas
      const canvas = canvasRef.current!;
      const ctx = canvas.getContext("2d")!;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw camera feed (mirrored)
      ctx.save();
      ctx.scale(-1, 1);
      ctx.drawImage(results.image, -canvas.width, 0, canvas.width, canvas.height);
      ctx.restore();

      if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        setHandDetected(true);
        const handLandmarks = results.multiHandLandmarks[0];

        // Draw landmarks
        drawLandmarks(ctx, handLandmarks, canvas.width, canvas.height);

        // Extract and send
        const landmarks = extractLandmarks(handLandmarks);
        onLandmarks(landmarks);
      } else {
        setHandDetected(false);
      }
    });

    // Start camera
    const camera = new Camera(videoRef.current, {
      onFrame: async () => {
        await hands.send({ image: videoRef.current! });
      },
      width: CAMERA_WIDTH,
      height: CAMERA_HEIGHT,
    });

    camera.start();
    cameraRef.current = camera;

    return () => {
      camera.stop();
      hands.close();
    };
  }, [enabled]);

  return { isLoading, handDetected };
}

// Draw hand landmarks on canvas
function drawLandmarks(
  ctx: CanvasRenderingContext2D,
  landmarks: any[],
  width: number,
  height: number
) {
  // Draw connections
  const connections = [
    [0,1],[1,2],[2,3],[3,4],       // Thumb
    [0,5],[5,6],[6,7],[7,8],       // Index
    [0,9],[9,10],[10,11],[11,12],  // Middle
    [0,13],[13,14],[14,15],[15,16],// Ring
    [0,17],[17,18],[18,19],[19,20],// Pinky
    [5,9],[9,13],[13,17]           // Palm
  ];

  ctx.strokeStyle = "#00FF00";
  ctx.lineWidth = 2;

  for (const [i, j] of connections) {
    const x1 = (1 - landmarks[i].x) * width;  // Mirrored
    const y1 = landmarks[i].y * height;
    const x2 = (1 - landmarks[j].x) * width;
    const y2 = landmarks[j].y * height;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  }

  // Draw points
  ctx.fillStyle = "#FF0000";
  for (const lm of landmarks) {
    const x = (1 - lm.x) * width;
    const y = lm.y * height;
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, 2 * Math.PI);
    ctx.fill();
  }
}
```

---

## Step 9: Main Recognition Page

### `src/pages/Recognize.tsx`

```tsx
import { useRef, useState, useCallback } from "react";
import { useMediaPipe } from "../hooks/useMediaPipe";
import { useWebSocket } from "../hooks/useWebSocket";
import { Language } from "../types";
import { CAMERA_WIDTH, CAMERA_HEIGHT } from "../utils/constants";

export default function Recognize() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [language, setLanguage] = useState<Language>("en");
  const [isRunning, setIsRunning] = useState(true);

  const { isConnected, lastResponse, sendLandmarks, sendClear } = useWebSocket();

  const handleLandmarks = useCallback(
    (landmarks: number[]) => {
      sendLandmarks(landmarks, language);
    },
    [sendLandmarks, language]
  );

  const { isLoading, handDetected } = useMediaPipe({
    videoRef,
    canvasRef,
    onLandmarks: handleLandmarks,
    enabled: isRunning,
  });

  const mode = lastResponse?.mode || "letter";
  const prediction = lastResponse?.prediction;
  const decoder = lastResponse?.decoder;

  return (
    <div className="min-h-screen bg-gray-900 text-white p-4">
      {/* Header */}
      <div className="flex justify-between items-center mb-4">
        <h1 className="text-2xl font-bold">ESHARA Recognition</h1>
        <div className="flex gap-2">
          {/* Language Toggle */}
          <button
            onClick={() => setLanguage(language === "en" ? "ar" : "en")}
            className="px-4 py-2 bg-indigo-600 rounded-lg hover:bg-indigo-700"
          >
            {language === "en" ? "العربية" : "English"}
          </button>
          {/* Start/Stop */}
          <button
            onClick={() => setIsRunning(!isRunning)}
            className={`px-4 py-2 rounded-lg ${
              isRunning ? "bg-red-600 hover:bg-red-700" : "bg-green-600 hover:bg-green-700"
            }`}
          >
            {isRunning ? "Stop" : "Start"}
          </button>
          {/* Clear */}
          <button
            onClick={sendClear}
            className="px-4 py-2 bg-gray-600 rounded-lg hover:bg-gray-700"
          >
            Clear
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Camera Feed */}
        <div className="lg:col-span-2 relative">
          <video ref={videoRef} className="hidden" />
          <canvas
            ref={canvasRef}
            width={CAMERA_WIDTH}
            height={CAMERA_HEIGHT}
            className="w-full rounded-xl border-2 border-gray-700"
          />

          {/* Status Badges */}
          <div className="absolute top-4 left-4 flex gap-2">
            {/* Connection */}
            <span className={`px-3 py-1 rounded-full text-sm ${
              isConnected ? "bg-green-600" : "bg-red-600"
            }`}>
              {isConnected ? "Connected" : "Disconnected"}
            </span>

            {/* Mode */}
            <span className={`px-3 py-1 rounded-full text-sm ${
              mode === "letter" ? "bg-blue-600" : "bg-purple-600"
            }`}>
              {mode === "letter" ? "📝 Letter Mode" : "🤟 Word Mode"}
            </span>

            {/* Hand Detection */}
            <span className={`px-3 py-1 rounded-full text-sm ${
              handDetected ? "bg-green-600" : "bg-yellow-600"
            }`}>
              {handDetected ? "✋ Hand Detected" : "No Hand"}
            </span>
          </div>

          {/* Loading Overlay */}
          {isLoading && (
            <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50 rounded-xl">
              <div className="text-center">
                <div className="animate-spin h-10 w-10 border-4 border-indigo-500 border-t-transparent rounded-full mx-auto mb-4" />
                <p>Loading MediaPipe...</p>
              </div>
            </div>
          )}
        </div>

        {/* Right Panel — Predictions & Text */}
        <div className="space-y-4">
          {/* Current Prediction */}
          <div className="bg-gray-800 rounded-xl p-6">
            <h2 className="text-lg font-semibold mb-2">Current Prediction</h2>
            {prediction ? (
              <div className="text-center">
                <p className="text-6xl font-bold mb-2">
                  {"letter" in prediction
                    ? (prediction as any).letter
                    : (prediction as any).word}
                </p>
                <div className="w-full bg-gray-700 rounded-full h-3 mb-2">
                  <div
                    className="bg-indigo-500 h-3 rounded-full transition-all"
                    style={{ width: `${(prediction.confidence * 100)}%` }}
                  />
                </div>
                <p className="text-sm text-gray-400">
                  {(prediction.confidence * 100).toFixed(1)}% confidence
                </p>
              </div>
            ) : (
              <p className="text-gray-500 text-center">Show a sign...</p>
            )}
          </div>

          {/* Built Text / Sentence */}
          <div className="bg-gray-800 rounded-xl p-6">
            <h2 className="text-lg font-semibold mb-2">
              {mode === "letter" ? "Built Text" : "Sentence"}
            </h2>
            <div className="min-h-[100px] bg-gray-900 rounded-lg p-4 text-xl"
                 dir={language === "ar" ? "rtl" : "ltr"}>
              {decoder?.text || decoder?.sentence || (
                <span className="text-gray-500">Start signing...</span>
              )}
            </div>
          </div>

          {/* Word Mode: Frame Progress */}
          {mode === "word" && lastResponse?.frames_buffered !== undefined && (
            <div className="bg-gray-800 rounded-xl p-4">
              <p className="text-sm text-gray-400 mb-2">Frame Buffer</p>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div
                  className="bg-purple-500 h-2 rounded-full transition-all"
                  style={{
                    width: `${(lastResponse.frames_buffered / (lastResponse.frames_needed || 30)) * 100}%`
                  }}
                />
              </div>
              <p className="text-xs text-gray-500 mt-1">
                {lastResponse.frames_buffered} / {lastResponse.frames_needed} frames
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
```

---

## Step 10: Home Page

### `src/pages/Home.tsx`

```tsx
import { Link } from "react-router-dom";

export default function Home() {
  return (
    <div className="min-h-screen bg-gray-900 text-white flex items-center justify-center">
      <div className="text-center max-w-2xl px-4">
        <h1 className="text-6xl font-bold mb-4">
          <span className="text-indigo-400">ESHARA</span>
        </h1>
        <p className="text-xl text-gray-300 mb-8">
          Sign Language Recognition System — Letters & Words
        </p>
        <p className="text-gray-400 mb-12">
          Bilingual support for ASL (English) and ArSL (Arabic).
          Real-time recognition using your webcam.
        </p>

        <Link
          to="/recognize"
          className="px-8 py-4 bg-indigo-600 text-white text-lg font-semibold rounded-xl hover:bg-indigo-700 transition-colors"
        >
          Start Recognition →
        </Link>

        <div className="mt-16 grid grid-cols-3 gap-8 text-center">
          <div>
            <p className="text-3xl font-bold text-indigo-400">29</p>
            <p className="text-gray-400">ASL Letters</p>
          </div>
          <div>
            <p className="text-3xl font-bold text-purple-400">157</p>
            <p className="text-gray-400">Words</p>
          </div>
          <div>
            <p className="text-3xl font-bold text-emerald-400">2</p>
            <p className="text-gray-400">Languages</p>
          </div>
        </div>
      </div>
    </div>
  );
}
```

---

## Step 11: App Router

### `src/App.tsx`

```tsx
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import Recognize from "./pages/Recognize";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/recognize" element={<Recognize />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
```

---

## Step 12: Environment Variables

### `web/.env`

```env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000/ws/recognize
```

---

## Step 13: Run the Web App

```powershell
cd "m:\Term 10\Grad\Deployment\web"

npm run dev

# Opens at http://localhost:5173
# Make sure backend is running on :8000
```

### What You Should See

1. **Home page** with ESHARA branding → Click "Start Recognition"
2. **Recognition page**:
   - Camera feed with green hand landmarks drawn
   - "Letter Mode" / "Word Mode" badge switches automatically
   - Current prediction shown with confidence bar
   - Built text/sentence accumulates as you sign
   - Language toggle (English ↔ Arabic)

---

## Internationalization (Arabic Support)

### `src/i18n/en.json`
```json
{
  "app_name": "ESHARA",
  "start": "Start Recognition",
  "stop": "Stop",
  "clear": "Clear",
  "language": "العربية",
  "letter_mode": "Letter Mode",
  "word_mode": "Word Mode",
  "built_text": "Built Text",
  "sentence": "Sentence",
  "confidence": "Confidence",
  "no_hand": "No Hand Detected",
  "hand_detected": "Hand Detected",
  "loading": "Loading MediaPipe..."
}
```

### `src/i18n/ar.json`
```json
{
  "app_name": "إشارة",
  "start": "ابدأ التعرف",
  "stop": "إيقاف",
  "clear": "مسح",
  "language": "English",
  "letter_mode": "وضع الحروف",
  "word_mode": "وضع الكلمات",
  "built_text": "النص المبني",
  "sentence": "الجملة",
  "confidence": "الثقة",
  "no_hand": "لم يتم اكتشاف يد",
  "hand_detected": "تم اكتشاف يد",
  "loading": "جاري تحميل MediaPipe..."
}
```

---

## Summary

| Step | What You Did | Estimated Time |
|------|-------------|---------------|
| 1 | Create React project | 5 min |
| 2 | Install deps | 5 min |
| 3 | Tailwind CSS setup | 10 min |
| 4-5 | Types + Constants | 10 min |
| 6 | Landmark utility | 15 min |
| 7-8 | WebSocket + MediaPipe hooks | 45 min |
| 9 | Recognition page (main) | 60 min |
| 10-11 | Home page + Router | 20 min |
| 12-13 | Env vars + Run | 5 min |
| i18n | Arabic translations | 15 min |
| **Total** | **Complete web frontend** | **~3-4 hours** |

> **Key concept**: MediaPipe runs IN THE BROWSER (client-side). Only the 63 landmark numbers are sent to the server.
> This means minimal server load and fast predictions.
