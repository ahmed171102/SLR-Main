const bcrypt = require("bcryptjs");
const prisma = require("../lib/prisma");
const { signAccessToken } = require("../utils/jwt");

function sanitizeUser(user) {
  return {
    id: user.id,
    name: user.name,
    email: user.email,
    createdAt: user.createdAt,
  };
}

async function register(req, res) {
  try {
    const { name, email, password } = req.body;

    const normalizedPassword = String(password || "").trim();

    if (!name || !email || !normalizedPassword) {
      return res.status(400).json({ message: "name, email, and password are required" });
    }

    if (normalizedPassword.length < 6) {
      return res.status(400).json({ message: "Password must be at least 6 characters" });
    }

    const normalizedEmail = email.trim().toLowerCase();
    const existingUser = await prisma.user.findUnique({ where: { email: normalizedEmail } });

    if (existingUser) {
      return res.status(409).json({ message: "Email already registered" });
    }

    const passwordHash = await bcrypt.hash(normalizedPassword, 10);
    const user = await prisma.user.create({
      data: {
        name: name.trim(),
        email: normalizedEmail,
        passwordHash,
      },
    });

    const token = signAccessToken({ userId: user.id, email: user.email });
    return res.status(201).json({ token, user: sanitizeUser(user) });
  } catch (error) {
    console.error("register error:", error);
    return res.status(500).json({ message: "Failed to register user" });
  }
}

async function login(req, res) {
  try {
    const { email, password } = req.body;

    const normalizedPassword = String(password || "").trim();

    if (!email || !normalizedPassword) {
      return res.status(400).json({ message: "email and password are required" });
    }

    const normalizedEmail = email.trim().toLowerCase();
    const user = await prisma.user.findUnique({ where: { email: normalizedEmail } });

    if (!user) {
      return res.status(401).json({ message: "Invalid email or password" });
    }

    const isValidPassword = await bcrypt.compare(normalizedPassword, user.passwordHash);
    if (!isValidPassword) {
      return res.status(401).json({ message: "Invalid email or password" });
    }

    const token = signAccessToken({ userId: user.id, email: user.email });
    return res.status(200).json({ token, user: sanitizeUser(user) });
  } catch (error) {
    console.error("login error:", error);
    return res.status(500).json({ message: "Failed to login user" });
  }
}

async function me(req, res) {
  return res.status(200).json({ user: req.user });
}

module.exports = {
  register,
  login,
  me,
};
