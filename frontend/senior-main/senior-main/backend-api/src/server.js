require("dotenv").config();

const http = require("http");
const { Server } = require("socket.io");

const app = require("./app");
const prisma = require("./lib/prisma");
const { verifyAccessToken } = require("./utils/jwt");
const { registerChatHandlers } = require("./sockets/chat.socket");

const PORT = process.env.PORT || 5000;
const CLIENT_ORIGIN = process.env.CLIENT_ORIGIN || "*";

const server = http.createServer(app);

const io = new Server(server, {
  cors: {
    origin: CLIENT_ORIGIN,
    credentials: true,
  },
});

io.use(async (socket, next) => {
  try {
    const authHeader = socket.handshake.headers.authorization || "";
    const bearerToken = authHeader.startsWith("Bearer ") ? authHeader.slice(7) : null;
    const handshakeToken = socket.handshake.auth?.token || null;
    const token = bearerToken || handshakeToken;

    if (!token) {
      return next(new Error("Missing access token"));
    }

    const decoded = verifyAccessToken(token);
    const user = await prisma.user.findUnique({
      where: { id: decoded.userId },
      select: { id: true, name: true, email: true },
    });

    if (!user) {
      return next(new Error("Invalid token user"));
    }

    socket.user = user;
    return next();
  } catch (_error) {
    return next(new Error("Invalid or expired token"));
  }
});

io.on("connection", (socket) => {
  console.log(`Socket connected: ${socket.id} (${socket.user.email})`);
  registerChatHandlers(io, socket);

  socket.on("disconnect", () => {
    console.log(`Socket disconnected: ${socket.id}`);
  });
});

server.listen(PORT, () => {
  console.log(`Backend API listening on http://localhost:${PORT}`);
});
