const router = require("express").Router();
const { requireAuth } = require("../middleware/auth.middleware");
const { listUsers } = require("../controllers/users.controller");

router.get("/", requireAuth, listUsers);

module.exports = router;
