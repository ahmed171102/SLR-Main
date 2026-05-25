const router = require("express").Router();
const { requireAuth } = require("../middleware/auth.middleware");
const { predict } = require("../controllers/predict.controller");

router.post("/", requireAuth, predict);

module.exports = router;
