# 02 — Tech Stack: Languages, Frameworks & Tools

> Everything you need to install, learn, and use for deployment.

---

## Languages You'll Write Code In

| Language | Where Used | Difficulty | You Know It? |
|----------|-----------|------------|-------------|
| **Python** | Backend API | Easy (you already use it) | YES (from notebooks) |
| **TypeScript/JavaScript** | Web frontend + Mobile app | Medium | Need to learn basics |
| **HTML/CSS** | Web UI styling | Easy | Minimal needed |
| **Dockerfile** | Container config | Easy (just a config file) | Copy-paste |

### Python (Backend) — You Already Know This

You use Python in every notebook. The backend is just Python organized into files instead of cells.

**New Python concepts to learn:**
- FastAPI framework (like Flask but faster) — 1 day to learn
- WebSocket connections — simple, FastAPI handles it
- `async/await` keywords — similar to regular Python, just add `async` before functions

### TypeScript (Web + Mobile) — New but Easy

TypeScript = JavaScript + type safety. If you know Python, you can learn TypeScript in 2-3 days.

**Key differences from Python:**
```typescript
// Python:  def greet(name: str) -> str:
// TypeScript:
function greet(name: string): string {
    return `Hello ${name}`;
}

// Python:  my_list = [1, 2, 3]
// TypeScript:
const myList: number[] = [1, 2, 3];

// Python:  my_dict = {"key": "value"}
// TypeScript:
const myDict: Record<string, string> = { key: "value" };
```

---

## Frameworks & Libraries

### Backend

| Package | Version | Purpose | Install |
|---------|---------|---------|---------|
| **FastAPI** | 0.104+ | Web framework (REST + WebSocket) | `pip install fastapi` |
| **Uvicorn** | 0.24+ | ASGI server (runs FastAPI) | `pip install uvicorn` |
| **TensorFlow** | 2.10.0 | Load & run .h5 models | `pip install tensorflow==2.10.0` |
| **MediaPipe** | 0.10.x | Hand detection (optional, for server-side) | `pip install mediapipe` |
| **NumPy** | 1.23+ | Array operations | `pip install numpy` |
| **Pandas** | 2.0+ | Load CSV label files | `pip install pandas` |
| **scikit-learn** | 1.3+ | LabelEncoder for classes | `pip install scikit-learn` |
| **python-dotenv** | 1.0+ | Load .env config | `pip install python-dotenv` |
| **websockets** | 12+ | WebSocket support | `pip install websockets` |

**One-line install:**
```bash
pip install fastapi uvicorn tensorflow==2.10.0 mediapipe numpy pandas scikit-learn python-dotenv websockets
```

### Web Frontend

| Package | Version | Purpose | Install |
|---------|---------|---------|---------|
| **React** | 18+ | UI framework | `npm create vite@latest` |
| **TypeScript** | 5+ | Type-safe JavaScript | Included with Vite |
| **Tailwind CSS** | 3+ | Utility CSS styling | `npm install tailwindcss` |
| **@mediapipe/hands** | 0.4+ | Hand detection in browser | `npm install @mediapipe/hands` |
| **@mediapipe/camera_utils** | 0.3+ | Camera access utility | `npm install @mediapipe/camera_utils` |
| **@mediapipe/drawing_utils** | 0.3+ | Draw landmarks on canvas | `npm install @mediapipe/drawing_utils` |
| **react-router-dom** | 6+ | Page routing | `npm install react-router-dom` |
| **i18next** | 23+ | Arabic/English translations | `npm install i18next react-i18next` |

**One-line install:**
```bash
npm create vite@latest eshara-web -- --template react-ts
cd eshara-web
npm install @mediapipe/hands @mediapipe/camera_utils @mediapipe/drawing_utils react-router-dom i18next react-i18next tailwindcss
```

### Mobile App

| Package | Version | Purpose | Install |
|---------|---------|---------|---------|
| **React Native** | 0.73+ | Mobile framework | Via Expo |
| **Expo** | 50+ | Dev tools + build | `npx create-expo-app` |
| **expo-camera** | 15+ | Camera access | `npx expo install expo-camera` |
| **@mediapipe/tasks-vision** | 0.10+ | Hand detection | `npm install @mediapipe/tasks-vision` |
| **react-native-tflite** | 1+ | Run .tflite models | `npm install react-native-tflite` |
| **@react-navigation/native** | 6+ | Screen navigation | `npm install @react-navigation/native` |
| **expo-file-system** | 16+ | Load model files | `npx expo install expo-file-system` |

**One-line install:**
```bash
npx create-expo-app eshara-mobile --template blank-typescript
cd eshara-mobile
npx expo install expo-camera expo-file-system
npm install @mediapipe/tasks-vision react-native-tflite @react-navigation/native
```

---

## Development Tools

### Required (Must Install)

| Tool | Purpose | Download |
|------|---------|----------|
| **Node.js 20 LTS** | Run JavaScript/TypeScript | https://nodejs.org |
| **Python 3.9** | Run backend | https://python.org (you have this) |
| **Git** | Version control | https://git-scm.com |
| **VS Code** | Code editor | You have this |

### Recommended VS Code Extensions

| Extension | Purpose |
|-----------|---------|
| Python | Python IntelliSense |
| ES7+ React/Redux snippets | React code snippets |
| Tailwind CSS IntelliSense | CSS autocomplete |
| Prettier | Code formatting |
| Thunder Client | Test API endpoints (like Postman) |
| Docker | Docker file support |

### Optional Tools

| Tool | Purpose | When Needed |
|------|---------|-------------|
| **Docker Desktop** | Run containers locally | Phase 5 (cloud deploy) |
| **Expo Go** (phone app) | Test mobile on real phone | Phase 4 (mobile) |
| **Postman** | Test API endpoints | Phase 2 (backend) |
| **Android Studio** | Android emulator | Phase 4 (if no phone) |

---

## Learning Resources (Quick Start)

| Topic | Best Resource | Time to Learn |
|-------|--------------|---------------|
| FastAPI basics | https://fastapi.tiangolo.com/tutorial/ | 2-3 hours |
| React basics | https://react.dev/learn | 1 day |
| TypeScript basics | https://www.typescriptlang.org/docs/handbook/ | 2-3 hours |
| Tailwind CSS | https://tailwindcss.com/docs | 1-2 hours |
| MediaPipe JS | https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/web_js | 1-2 hours |
| React Native + Expo | https://docs.expo.dev/tutorial/introduction/ | 1 day |
| Docker basics | https://docs.docker.com/get-started/ | 2-3 hours |
| WebSocket concept | https://fastapi.tiangolo.com/advanced/websockets/ | 30 min |

> **Total new learning**: ~3-4 days if you focus. Most of this is watching tutorials while coding along.
