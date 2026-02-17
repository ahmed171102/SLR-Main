# 07 — Mobile App: Step-by-Step Build Guide

> Build a React Native + Expo mobile app with on-device inference (100% offline capable).

---

## Overview

The mobile app is different from the web app:
- **Web**: sends landmarks to server → server runs model → returns prediction
- **Mobile**: runs EVERYTHING on-device → camera + MediaPipe + TFLite model → instant result

```
Camera → MediaPipe (on-device) → TFLite model (on-device) → Result
                    No internet needed!
```

**Why on-device?**
- Works offline (no WiFi needed)
- Instant predictions (no network latency)
- User data never leaves their phone
- No server costs for mobile users

---

## Prerequisites

- Node.js 20 LTS installed
- TFLite model files ready (see `08_MODEL_CONVERSION.md`)
- Expo Go app on your phone (for testing) — download from App Store / Play Store
- Android Studio (optional, for emulator)

---

## Step 1: Create Expo Project

```powershell
cd "m:\Term 10\Grad\Deployment"

# Create Expo project
npx create-expo-app mobile --template blank-typescript

cd mobile
```

---

## Step 2: Install Dependencies

```powershell
cd "m:\Term 10\Grad\Deployment\mobile"

# Camera
npx expo install expo-camera

# File system (for loading models)
npx expo install expo-file-system

# Navigation
npm install @react-navigation/native @react-navigation/native-stack
npx expo install react-native-screens react-native-safe-area-context

# TFLite inference
npm install react-native-tflite

# MediaPipe for hand detection
npm install @mediapipe/tasks-vision

# Localization
npm install i18next react-i18next

# Asset loading
npx expo install expo-asset
```

---

## Step 3: Copy Model Files

Place your TFLite models in `mobile/assets/models/`:

```
mobile/assets/models/
├── asl_letter_model.tflite        # Converted from asl_mediapipe_mlp_model.h5
├── arsl_letter_model.tflite       # Converted from arsl_mediapipe_mlp_model_final.h5
├── asl_word_model.tflite          # Converted from asl_word_lstm_model_best.h5
├── asl_letter_labels.json         # ["A", "B", "C", ..., "space", "del", "nothing"]
├── arsl_letter_labels.json        # ["أ", "ب", "ت", ...]
└── word_labels.json               # [{id: 0, en: "drink", ar: "يشرب"}, ...]
```

> See `08_MODEL_CONVERSION.md` for how to create these files.

---

## Step 4: Configure App

### `app.json`

```json
{
  "expo": {
    "name": "ESHARA",
    "slug": "eshara",
    "version": "1.0.0",
    "orientation": "portrait",
    "icon": "./assets/icon.png",
    "splash": {
      "image": "./assets/splash.png",
      "resizeMode": "contain",
      "backgroundColor": "#1f2937"
    },
    "ios": {
      "supportsTablet": true,
      "bundleIdentifier": "com.eshara.app",
      "infoPlist": {
        "NSCameraUsageDescription": "ESHARA needs camera access to recognize sign language."
      }
    },
    "android": {
      "adaptiveIcon": {
        "foregroundImage": "./assets/adaptive-icon.png",
        "backgroundColor": "#1f2937"
      },
      "package": "com.eshara.app",
      "permissions": ["CAMERA"]
    },
    "plugins": [
      [
        "expo-camera",
        {
          "cameraPermission": "ESHARA needs camera access to recognize sign language gestures."
        }
      ]
    ]
  }
}
```

---

## Step 5: TFLite Model Service

### `src/services/tfliteModel.ts`

```typescript
import { loadTensorflowModel, TensorflowModel } from "react-native-tflite";
import * as FileSystem from "expo-file-system";
import { Asset } from "expo-asset";

// Model instances
let aslLetterModel: TensorflowModel | null = null;
let arslLetterModel: TensorflowModel | null = null;
let aslWordModel: TensorflowModel | null = null;

// Label data
let aslLetterLabels: string[] = [];
let arslLetterLabels: string[] = [];
let wordLabels: { id: number; en: string; ar: string }[] = [];

/**
 * Load all TFLite models and label files.
 * Call once on app startup.
 */
export async function loadModels(): Promise<void> {
  console.log("Loading TFLite models...");

  // Load models
  aslLetterModel = await loadTensorflowModel(
    require("../../assets/models/asl_letter_model.tflite")
  );
  console.log("  ✓ ASL Letter model loaded");

  arslLetterModel = await loadTensorflowModel(
    require("../../assets/models/arsl_letter_model.tflite")
  );
  console.log("  ✓ ArSL Letter model loaded");

  aslWordModel = await loadTensorflowModel(
    require("../../assets/models/asl_word_model.tflite")
  );
  console.log("  ✓ ASL Word model loaded");

  // Load labels
  const loadJson = async (asset: any) => {
    const [{ localUri }] = await Asset.loadAsync(asset);
    const content = await FileSystem.readAsStringAsync(localUri!);
    return JSON.parse(content);
  };

  aslLetterLabels = await loadJson(
    require("../../assets/models/asl_letter_labels.json")
  );
  arslLetterLabels = await loadJson(
    require("../../assets/models/arsl_letter_labels.json")
  );
  wordLabels = await loadJson(
    require("../../assets/models/word_labels.json")
  );

  console.log("All models loaded!");
}

/**
 * Predict a letter from 63 landmark floats.
 */
export function predictLetter(
  landmarks: number[],
  language: "en" | "ar" = "en"
): { letter: string; confidence: number } | null {
  const model = language === "ar" ? arslLetterModel : aslLetterModel;
  const labels = language === "ar" ? arslLetterLabels : aslLetterLabels;

  if (!model || labels.length === 0) return null;

  // Run inference: input shape (1, 63)
  const input = new Float32Array(landmarks);
  const output = model.runSync([input]);
  const predictions = output[0] as Float32Array;

  // Find max
  let maxIdx = 0;
  let maxVal = predictions[0];
  for (let i = 1; i < predictions.length; i++) {
    if (predictions[i] > maxVal) {
      maxVal = predictions[i];
      maxIdx = i;
    }
  }

  return {
    letter: labels[maxIdx],
    confidence: maxVal,
  };
}

/**
 * Predict a word from 30 frames of 63 floats each.
 */
export function predictWord(
  frames: number[][],
  language: "en" | "ar" = "en"
): { word: string; wordEn: string; wordAr: string; confidence: number } | null {
  if (!aslWordModel || wordLabels.length === 0) return null;
  if (frames.length !== 30) return null;

  // Flatten: 30 × 63 = 1890 floats
  const input = new Float32Array(frames.flat());
  const output = aslWordModel.runSync([input]);
  const predictions = output[0] as Float32Array;

  // Find max
  let maxIdx = 0;
  let maxVal = predictions[0];
  for (let i = 1; i < predictions.length; i++) {
    if (predictions[i] > maxVal) {
      maxVal = predictions[i];
      maxIdx = i;
    }
  }

  const label = wordLabels[maxIdx] || { en: "unknown", ar: "" };

  return {
    word: language === "ar" ? label.ar : label.en,
    wordEn: label.en,
    wordAr: label.ar,
    confidence: maxVal,
  };
}
```

---

## Step 6: Letter & Word Decoders (JavaScript Port)

### `src/services/letterDecoder.ts`

```typescript
/**
 * JavaScript port of letter_stream_decoder.py.
 * Converts per-frame letter predictions into text with stabilization.
 */

const STABLE_WINDOW = 5;
const MAJORITY_RATIO = 0.7;
const COOLDOWN_MS = 600;
const CONFIDENCE_THRESHOLD = 0.85;

export class LetterDecoder {
  private history: string[] = [];
  private text = "";
  private currentWord = "";
  private lastCommitted: string | null = null;
  private lastCommitTime = 0;

  update(label: string, confidence: number): {
    committed: boolean;
    text: string;
    currentWord: string;
    event: string;
  } {
    const now = Date.now();

    if (confidence < CONFIDENCE_THRESHOLD || label === "nothing") {
      this.history = [];
      return this.result(false, "none");
    }

    this.history.push(label);
    if (this.history.length > STABLE_WINDOW) {
      this.history.shift();
    }

    if (this.history.length >= STABLE_WINDOW) {
      // Count occurrences
      const counts = new Map<string, number>();
      for (const l of this.history) {
        counts.set(l, (counts.get(l) || 0) + 1);
      }

      let mostCommon = "";
      let maxCount = 0;
      for (const [l, c] of counts) {
        if (c > maxCount) { maxCount = c; mostCommon = l; }
      }

      if (maxCount / this.history.length >= MAJORITY_RATIO) {
        // Cooldown check
        if (mostCommon === this.lastCommitted && now - this.lastCommitTime < COOLDOWN_MS) {
          return this.result(false, "cooldown");
        }

        this.lastCommitted = mostCommon;
        this.lastCommitTime = now;
        this.history = [];

        if (mostCommon === "space") {
          this.text += this.currentWord + " ";
          this.currentWord = "";
          return this.result(true, "space");
        } else if (mostCommon === "del") {
          if (this.currentWord) {
            this.currentWord = this.currentWord.slice(0, -1);
          }
          return this.result(true, "delete");
        } else {
          this.currentWord += mostCommon;
          return this.result(true, "letter_added");
        }
      }
    }

    return this.result(false, "none");
  }

  private result(committed: boolean, event: string) {
    return {
      committed,
      text: this.text + this.currentWord,
      currentWord: this.currentWord,
      event,
    };
  }

  clear() {
    this.history = [];
    this.text = "";
    this.currentWord = "";
    this.lastCommitted = null;
    this.lastCommitTime = 0;
  }
}
```

### `src/services/wordDecoder.ts`

```typescript
/**
 * Word sentence builder with stabilization and cooldown.
 */

const STABILITY_WINDOW = 3;
const COOLDOWN_MS = 2000;
const CONFIDENCE_THRESHOLD = 0.35;

export class WordDecoder {
  private history: string[] = [];
  private words: string[] = [];
  private lastCommitted: string | null = null;
  private lastCommitTime = 0;

  update(word: string, confidence: number): {
    committed: boolean;
    sentence: string;
    lastWord: string;
    event: string;
  } {
    const now = Date.now();

    if (confidence < CONFIDENCE_THRESHOLD) {
      this.history = [];
      return this.result(false, "low_confidence");
    }

    this.history.push(word);
    if (this.history.length > STABILITY_WINDOW) {
      this.history.shift();
    }

    if (this.history.length >= STABILITY_WINDOW) {
      const allSame = this.history.every((w) => w === this.history[0]);
      if (allSame) {
        const stableWord = this.history[0];

        if (stableWord === this.lastCommitted && now - this.lastCommitTime < COOLDOWN_MS) {
          return this.result(false, "cooldown");
        }

        this.words.push(stableWord);
        this.lastCommitted = stableWord;
        this.lastCommitTime = now;
        this.history = [];

        return this.result(true, "word_added");
      }
    }

    return this.result(false, "none");
  }

  private result(committed: boolean, event: string) {
    return {
      committed,
      sentence: this.words.join(" "),
      lastWord: this.words[this.words.length - 1] || "",
      event,
    };
  }

  clear() {
    this.history = [];
    this.words = [];
    this.lastCommitted = null;
    this.lastCommitTime = 0;
  }
}
```

---

## Step 7: Main Recognition Screen

### `src/screens/RecognizeScreen.tsx`

```tsx
import React, { useEffect, useState, useRef } from "react";
import { View, Text, StyleSheet, TouchableOpacity, Alert } from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";
import { predictLetter, predictWord, loadModels } from "../services/tfliteModel";
import { LetterDecoder } from "../services/letterDecoder";
import { WordDecoder } from "../services/wordDecoder";

type Mode = "letter" | "word";
type Language = "en" | "ar";

export default function RecognizeScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const [isReady, setIsReady] = useState(false);
  const [mode, setMode] = useState<Mode>("letter");
  const [language, setLanguage] = useState<Language>("en");
  const [prediction, setPrediction] = useState("");
  const [confidence, setConfidence] = useState(0);
  const [builtText, setBuiltText] = useState("");

  const letterDecoder = useRef(new LetterDecoder());
  const wordDecoder = useRef(new WordDecoder());
  const frameBuffer = useRef<number[][]>([]);

  // Load models on mount
  useEffect(() => {
    loadModels()
      .then(() => setIsReady(true))
      .catch((err) => Alert.alert("Error", "Failed to load models: " + err.message));
  }, []);

  // Request camera permission
  useEffect(() => {
    if (!permission?.granted) {
      requestPermission();
    }
  }, [permission]);

  if (!permission?.granted) {
    return (
      <View style={styles.container}>
        <Text style={styles.text}>Camera permission required</Text>
        <TouchableOpacity style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  if (!isReady) {
    return (
      <View style={styles.container}>
        <Text style={styles.text}>Loading models...</Text>
      </View>
    );
  }

  /**
   * NOTE: In a real implementation, you would process camera frames
   * through MediaPipe Hands to get landmarks, then feed those to
   * the TFLite model. The exact frame processing integration depends
   * on the MediaPipe React Native API available.
   *
   * Pseudo-flow:
   * 1. Camera frame captured
   * 2. MediaPipe extracts 21 hand landmarks
   * 3. Convert to 63 floats (x, y, z relative to wrist)
   * 4. Feed to predictLetter() or predictWord()
   * 5. Update UI with result
   */

  const handleClear = () => {
    letterDecoder.current.clear();
    wordDecoder.current.clear();
    frameBuffer.current = [];
    setBuiltText("");
    setPrediction("");
    setConfidence(0);
  };

  return (
    <View style={styles.container}>
      {/* Camera */}
      <CameraView style={styles.camera} facing="front">
        {/* Overlay UI */}
        <View style={styles.overlay}>
          {/* Top Bar */}
          <View style={styles.topBar}>
            <View style={[styles.badge, mode === "letter" ? styles.badgeLetter : styles.badgeWord]}>
              <Text style={styles.badgeText}>
                {mode === "letter" ? "📝 Letters" : "🤟 Words"}
              </Text>
            </View>
          </View>

          {/* Bottom Panel */}
          <View style={styles.bottomPanel}>
            {/* Prediction */}
            <Text style={styles.predictionText}>{prediction || "..."}</Text>
            <Text style={styles.confidenceText}>
              {confidence > 0 ? `${(confidence * 100).toFixed(0)}%` : ""}
            </Text>

            {/* Built Text */}
            <View style={styles.textBox}>
              <Text style={[styles.builtText, language === "ar" && { textAlign: "right" }]}>
                {builtText || "Start signing..."}
              </Text>
            </View>

            {/* Controls */}
            <View style={styles.controls}>
              <TouchableOpacity
                style={styles.controlButton}
                onPress={() => setMode(mode === "letter" ? "word" : "letter")}
              >
                <Text style={styles.buttonText}>
                  {mode === "letter" ? "Switch to Words" : "Switch to Letters"}
                </Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.controlButton}
                onPress={() => setLanguage(language === "en" ? "ar" : "en")}
              >
                <Text style={styles.buttonText}>
                  {language === "en" ? "العربية" : "English"}
                </Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={[styles.controlButton, { backgroundColor: "#ef4444" }]}
                onPress={handleClear}
              >
                <Text style={styles.buttonText}>Clear</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </CameraView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#111827" },
  camera: { flex: 1 },
  overlay: { flex: 1, justifyContent: "space-between" },
  topBar: { flexDirection: "row", padding: 16, paddingTop: 50 },
  badge: { paddingHorizontal: 12, paddingVertical: 6, borderRadius: 20 },
  badgeLetter: { backgroundColor: "#3b82f6" },
  badgeWord: { backgroundColor: "#8b5cf6" },
  badgeText: { color: "#fff", fontWeight: "bold", fontSize: 14 },
  bottomPanel: {
    backgroundColor: "rgba(0,0,0,0.8)",
    padding: 20,
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
  },
  predictionText: {
    color: "#fff",
    fontSize: 48,
    fontWeight: "bold",
    textAlign: "center",
  },
  confidenceText: {
    color: "#9ca3af",
    fontSize: 16,
    textAlign: "center",
    marginBottom: 12,
  },
  textBox: {
    backgroundColor: "#1f2937",
    padding: 16,
    borderRadius: 12,
    minHeight: 60,
    marginBottom: 12,
  },
  builtText: { color: "#fff", fontSize: 18 },
  controls: {
    flexDirection: "row",
    justifyContent: "space-between",
    gap: 8,
  },
  controlButton: {
    flex: 1,
    backgroundColor: "#4f46e5",
    paddingVertical: 12,
    borderRadius: 12,
    alignItems: "center",
  },
  button: {
    backgroundColor: "#4f46e5",
    paddingVertical: 16,
    paddingHorizontal: 32,
    borderRadius: 12,
    marginTop: 20,
  },
  buttonText: { color: "#fff", fontWeight: "bold", fontSize: 14 },
  text: { color: "#fff", fontSize: 18, textAlign: "center" },
});
```

---

## Step 8: Home Screen

### `src/screens/HomeScreen.tsx`

```tsx
import React from "react";
import { View, Text, StyleSheet, TouchableOpacity } from "react-native";

export default function HomeScreen({ navigation }: any) {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>ESHARA</Text>
      <Text style={styles.subtitle}>Sign Language Recognition</Text>
      <Text style={styles.description}>
        Letters & Words • English & Arabic{"\n"}
        100% On-Device • No Internet Needed
      </Text>

      <TouchableOpacity
        style={styles.button}
        onPress={() => navigation.navigate("Recognize")}
      >
        <Text style={styles.buttonText}>Start Recognition →</Text>
      </TouchableOpacity>

      <View style={styles.stats}>
        <View style={styles.stat}>
          <Text style={styles.statNumber}>29</Text>
          <Text style={styles.statLabel}>Letters</Text>
        </View>
        <View style={styles.stat}>
          <Text style={styles.statNumber}>157</Text>
          <Text style={styles.statLabel}>Words</Text>
        </View>
        <View style={styles.stat}>
          <Text style={styles.statNumber}>2</Text>
          <Text style={styles.statLabel}>Languages</Text>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#111827",
    justifyContent: "center",
    alignItems: "center",
    padding: 24,
  },
  title: { color: "#818cf8", fontSize: 48, fontWeight: "bold" },
  subtitle: { color: "#fff", fontSize: 20, marginTop: 8, marginBottom: 16 },
  description: { color: "#9ca3af", fontSize: 16, textAlign: "center", marginBottom: 40 },
  button: {
    backgroundColor: "#4f46e5",
    paddingVertical: 16,
    paddingHorizontal: 40,
    borderRadius: 16,
  },
  buttonText: { color: "#fff", fontSize: 18, fontWeight: "bold" },
  stats: { flexDirection: "row", marginTop: 60, gap: 40 },
  stat: { alignItems: "center" },
  statNumber: { color: "#818cf8", fontSize: 28, fontWeight: "bold" },
  statLabel: { color: "#9ca3af", fontSize: 14, marginTop: 4 },
});
```

---

## Step 9: App Entry Point

### `App.tsx`

```tsx
import React from "react";
import { NavigationContainer } from "@react-navigation/native";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import HomeScreen from "./src/screens/HomeScreen";
import RecognizeScreen from "./src/screens/RecognizeScreen";

const Stack = createNativeStackNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator
        screenOptions={{
          headerShown: false,
          contentStyle: { backgroundColor: "#111827" },
        }}
      >
        <Stack.Screen name="Home" component={HomeScreen} />
        <Stack.Screen name="Recognize" component={RecognizeScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
```

---

## Step 10: Test on Phone

```powershell
cd "m:\Term 10\Grad\Deployment\mobile"

# Start dev server
npx expo start

# Scan the QR code with Expo Go app on your phone
# Or press 'a' for Android emulator
```

---

## Important Notes

### MediaPipe on React Native
MediaPipe's hand detection in React Native is more complex than on web. Options:
1. **@mediapipe/tasks-vision** — May need a custom native module bridge
2. **expo-camera** frame processor — Use `expo-camera`'s `onFrame` callback
3. **react-native-vision-camera** + **react-native-worklets** — More advanced but better frame access

The recommended production approach in 2026:
- Use `react-native-vision-camera` v4+ with frame processors
- Process each frame through MediaPipe's hand landmarker
- This gives you the 63 floats needed for TFLite prediction

### Offline-First Architecture
The mobile app runs 100% offline:
- Models are bundled in the APK/IPA (adds ~10-20MB to app size)
- No API calls needed for basic recognition
- Internet only needed for: updates, history sync, cloud backup

---

## Summary

| Step | What You Did | Estimated Time |
|------|-------------|---------------|
| 1 | Create Expo project | 5 min |
| 2 | Install deps | 10 min |
| 3 | Copy model files | 10 min |
| 4 | Configure app.json | 15 min |
| 5 | TFLite model service | 45 min |
| 6 | Decoders (JS ports) | 30 min |
| 7 | Recognition screen | 60 min |
| 8-9 | Home screen + navigation | 20 min |
| 10 | Test on phone | 15 min |
| MediaPipe integration | Frame processing setup | 2-3 hours |
| **Total** | **Complete mobile app** | **~5-7 hours** |

> Mobile is the hardest frontend piece because of MediaPipe + TFLite integration on native.
> Start with the web version first, then port to mobile.
