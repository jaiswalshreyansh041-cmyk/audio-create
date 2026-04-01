export default function handler(req, res) {
  res.json({ GEMINI_API_KEY: process.env.GEMINI_API_KEY || process.env.API_KEY });
}
