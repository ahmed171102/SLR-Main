async function predict(req, res) {
  try {
    const { language, mode, landmarks } = req.body;

    if (!language || !mode || !landmarks) {
      return res.status(400).json({
        message: "language, mode, and landmarks are required",
      });
    }

    const modelServiceUrl = process.env.MODEL_SERVICE_URL || "http://127.0.0.1:8001";

    try {
      const response = await fetch(`${modelServiceUrl}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ language, mode, landmarks }),
      });

      if (!response.ok) {
        throw new Error(`Model service responded with ${response.status}`);
      }

      const result = await response.json();
      return res.status(200).json(result);
    } catch (_serviceError) {
      // Keep frontend flow unblocked even if model-service is temporarily unavailable.
      return res.status(200).json({
        prediction: {
          text: mode === "letters" ? (language === "ar" ? "ا" : "A") : language === "ar" ? "مرحبا" : "hello",
          confidence: 0.0,
          source: "fallback",
        },
      });
    }
  } catch (error) {
    console.error("predict error:", error);
    return res.status(500).json({ message: "Failed to process prediction request" });
  }
}

module.exports = {
  predict,
};
