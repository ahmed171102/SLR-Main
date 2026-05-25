const prisma = require("../lib/prisma");

async function listUsers(req, res) {
  try {
    const users = await prisma.user.findMany({
      where: {
        id: { not: req.user.id },
      },
      select: {
        id: true,
        name: true,
        email: true,
      },
      orderBy: {
        createdAt: "desc",
      },
    });

    return res.status(200).json({ users });
  } catch (error) {
    console.error("list users error:", error);
    return res.status(500).json({ message: "Failed to list users" });
  }
}

module.exports = {
  listUsers,
};
