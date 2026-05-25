const express = require("express");
const cors = require("cors");
const helmet = require("helmet");
const morgan = require("morgan");

const healthRouter = require("./routes/health.routes");
const authRouter = require("./routes/auth.routes");
const predictRouter = require("./routes/predict.routes");
const usersRouter = require("./routes/users.routes");

const app = express();

app.use(helmet());
app.use(cors());
app.use(morgan("dev"));
app.use(express.json({ limit: "10mb" }));

app.use("/health", healthRouter);
app.use("/auth", authRouter);
app.use("/predict", predictRouter);
app.use("/users", usersRouter);

app.get("/", (_req, res) => {
  res.json({
    message: "Backend API is running",
  });
});

module.exports = app;
