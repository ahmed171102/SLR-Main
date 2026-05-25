const prisma = require("../lib/prisma");

async function findOrCreateDirectRoom(userAId, userBId) {
  const memberships = await prisma.roomMember.findMany({
    where: {
      userId: { in: [userAId, userBId] },
      room: { type: "DIRECT" },
    },
    select: { roomId: true, userId: true },
  });

  const roomMembershipMap = new Map();
  for (const membership of memberships) {
    if (!roomMembershipMap.has(membership.roomId)) {
      roomMembershipMap.set(membership.roomId, new Set());
    }
    roomMembershipMap.get(membership.roomId).add(membership.userId);
  }

  for (const [roomId, users] of roomMembershipMap) {
    if (users.has(userAId) && users.has(userBId)) {
      return prisma.room.findUnique({ where: { id: roomId } });
    }
  }

  return prisma.room.create({
    data: {
      type: "DIRECT",
      members: {
        create: [{ userId: userAId }, { userId: userBId }],
      },
    },
  });
}

async function ensureUserInRoom(roomId, userId) {
  const membership = await prisma.roomMember.findUnique({
    where: { roomId_userId: { roomId, userId } },
  });
  return Boolean(membership);
}

function registerChatHandlers(io, socket) {
  socket.on("chat:join", async (payload = {}, ack) => {
    try {
      const { roomId, targetUserId } = payload;

      let resolvedRoomId = roomId;
      if (!resolvedRoomId && targetUserId) {
        const room = await findOrCreateDirectRoom(socket.user.id, targetUserId);
        resolvedRoomId = room.id;
      }

      if (!resolvedRoomId) {
        return ack?.({ ok: false, message: "roomId or targetUserId is required" });
      }

      const isMember = await ensureUserInRoom(resolvedRoomId, socket.user.id);
      if (!isMember) {
        return ack?.({ ok: false, message: "You are not a member of this room" });
      }

      socket.join(resolvedRoomId);
      const recentMessages = await prisma.message.findMany({
        where: { roomId: resolvedRoomId },
        orderBy: { createdAt: "asc" },
        take: 50,
        include: {
          sender: { select: { id: true, name: true, email: true } },
        },
      });

      return ack?.({
        ok: true,
        roomId: resolvedRoomId,
        messages: recentMessages.map((item) => ({
          id: item.id,
          roomId: item.roomId,
          content: item.content,
          contentType: item.contentType,
          createdAt: item.createdAt,
          sender: item.sender,
        })),
      });
    } catch (error) {
      console.error("chat:join error", error);
      return ack?.({ ok: false, message: "Failed to join room" });
    }
  });

  socket.on("chat:send", async (payload = {}, ack) => {
    try {
      const { roomId, content, contentType = "TEXT" } = payload;

      if (!roomId || !content || !String(content).trim()) {
        return ack?.({ ok: false, message: "roomId and content are required" });
      }

      const isMember = await ensureUserInRoom(roomId, socket.user.id);
      if (!isMember) {
        return ack?.({ ok: false, message: "You are not a member of this room" });
      }

      const message = await prisma.message.create({
        data: {
          roomId,
          senderId: socket.user.id,
          content: String(content).trim(),
          contentType,
        },
        include: {
          sender: { select: { id: true, name: true, email: true } },
        },
      });

      const output = {
        id: message.id,
        roomId: message.roomId,
        content: message.content,
        contentType: message.contentType,
        createdAt: message.createdAt,
        sender: message.sender,
      };

      io.to(roomId).emit("chat:receive", output);
      return ack?.({ ok: true, message: output });
    } catch (error) {
      console.error("chat:send error", error);
      return ack?.({ ok: false, message: "Failed to send message" });
    }
  });
}

module.exports = {
  registerChatHandlers,
};
