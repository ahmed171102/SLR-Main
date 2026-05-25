const router = require("express").Router();

router.get("/", (_req, res) => {
  res.status(200).json({
    status: "ok",
    service: "backend-api",
    time: new Date().toISOString(),
  });
});

module.exports = router;
