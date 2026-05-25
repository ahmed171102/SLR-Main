const jwt = require("jsonwebtoken");

function signAccessToken(payload) {
  const secret = process.env.JWT_SECRET;

  if (!secret) {
    throw new Error("JWT_SECRET is not configured");
  }

  return jwt.sign(payload, secret, { expiresIn: "7d" });
}

function verifyAccessToken(token) {
  const secret = process.env.JWT_SECRET;

  if (!secret) {
    throw new Error("JWT_SECRET is not configured");
  }

  return jwt.verify(token, secret);
}

module.exports = {
  signAccessToken,
  verifyAccessToken,
};
