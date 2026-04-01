export default function handler(req, res) {
  res.json({ 
    GEMINI_API_KEY: process.env.GEMINI_API_KEY || process.env.API_KEY,
    TRANSCRIPT_API_KEY: process.env.TRANSCRIPT_API_KEY,
    JSON_API_KEY: process.env.JSON_API_KEY
  });
}
