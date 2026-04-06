/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef } from 'react';
import { Upload, FileAudio, Activity, AlertTriangle, CheckCircle, BrainCircuit, Settings2, Download, Shield, MessageSquare } from 'lucide-react';
import { GoogleGenAI } from '@google/genai';

interface AudioStats {
  format: string;
  sampleRate: number;
  numChannels: number;
  duration: number;
  peakDB: number;
  rmsDB: number;
  clipCount: number;
  dcOffset: number;
  noiseLevelDB: number;
  snrDB: number;
  silenceRatio: number;
  maxSilenceDuration: number;
  rmsDynamicRange: number;
}

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [stats, setStats] = useState<AudioStats | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const [aiResult, setAiResult] = useState<string | null>(null);
  const [aiJsonResult, setAiJsonResult] = useState<string | null>(null);
  const [isAiAnalyzing, setIsAiAnalyzing] = useState(false);
  const [isGeneratingJson, setIsGeneratingJson] = useState(false);
  const [aiError, setAiError] = useState<string | null>(null);
  const noiseThreshold = -45;
  const snrThreshold = 20;
  const maxSilenceRatio = 20;
  const expectedSpeakers = 2;
  const minDuration = 30;
  const maxDuration = 300;

  const audioRef = useRef<HTMLAudioElement>(null);
  const audioBufferRef = useRef<AudioBuffer | null>(null);

  let parsedAiData: any = null;
  if (aiJsonResult) {
    try {
      parsedAiData = JSON.parse(aiJsonResult);
    } catch (e) {
      console.error("Failed to parse AI JSON result", e);
    }
  }

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;

    setFile(selectedFile);
    setStats(null);
    setError(null);
    setAiResult(null);
    setAiJsonResult(null);
    setAiError(null);
    setIsAnalyzing(true);

    try {
      const arrayBuffer = await selectedFile.arrayBuffer();
      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
      audioBufferRef.current = audioBuffer;

      const numChannels = audioBuffer.numberOfChannels;
      const sampleRate = audioBuffer.sampleRate;
      const duration = audioBuffer.duration;

      // Mixdown to mono for easier analysis
      const monoData = new Float32Array(audioBuffer.length);
      for (let i = 0; i < audioBuffer.length; i++) {
        let sum = 0;
        for (let c = 0; c < numChannels; c++) {
          sum += audioBuffer.getChannelData(c)[i];
        }
        monoData[i] = sum / numChannels;
      }

      let peak = 0;
      let sumSquares = 0;
      let clipCount = 0;
      let dcOffsetSum = 0;
      let silenceSamples = 0;
      const silenceThreshold = 0.001; // approx -60 dBFS

      let currentSilenceSamples = 0;
      let maxSilenceSamples = 0;
      let consecutiveClipSamples = 0; // true clipping = flat top across 3+ consecutive samples

      const frameSize = Math.floor(sampleRate * 0.05); // 50ms frames
      const rmsValues: number[] = [];
      
      const windowSize = sampleRate; // 1 second window for dynamic range
      let windowRmsSum = 0;
      let windowSamples = 0;
      const rmsWindows: number[] = [];

      for (let i = 0; i < monoData.length; i++) {
        const sample = monoData[i];
        const absSample = Math.abs(sample);

        if (absSample > peak) peak = absSample;
        sumSquares += sample * sample;
        dcOffsetSum += sample;

        // 1. Clipping detection — require 3 consecutive samples at >= 0.999
        // Single samples at 0.999 are codec artifacts, not real clipping
        if (absSample >= 0.999) {
          consecutiveClipSamples++;
          if (consecutiveClipSamples === 3) clipCount++; // count each clipping event once
        } else {
          consecutiveClipSamples = 0;
        }

        if (absSample < silenceThreshold) {
          silenceSamples++;
          currentSilenceSamples++;
          if (currentSilenceSamples > maxSilenceSamples) {
            maxSilenceSamples = currentSilenceSamples;
          }
        } else {
          currentSilenceSamples = 0;
        }

        windowRmsSum += sample * sample;
        windowSamples++;
        if (windowSamples >= windowSize) {
          const windowRms = Math.sqrt(windowRmsSum / windowSize);
          if (windowRms > 0.001) {
            rmsWindows.push(20 * Math.log10(windowRms));
          }
          windowRmsSum = 0;
          windowSamples = 0;
        }
      }

      for (let i = 0; i < monoData.length; i += frameSize) {
        let frameSumSq = 0;
        let count = 0;
        for (let j = 0; j < frameSize && i + j < monoData.length; j++) {
          frameSumSq += monoData[i + j] * monoData[i + j];
          count++;
        }
        rmsValues.push(Math.sqrt(frameSumSq / count));
      }

      rmsValues.sort((a, b) => a - b);
      // 2. Background noise level (10th percentile of frame RMS)
      const noiseRMS = rmsValues[Math.floor(rmsValues.length * 0.1)] || 0.00001;
      // Signal level (90th percentile of frame RMS)
      const signalRMS = rmsValues[Math.floor(rmsValues.length * 0.9)] || 0.00001;

      const noiseLevelDB = 20 * Math.log10(noiseRMS);
      const signalLevelDB = 20 * Math.log10(signalRMS);
      
      // 3. Signal-to-Noise Ratio
      const snrDB = signalLevelDB - noiseLevelDB;

      // 4. Silence ratio
      const silenceRatio = silenceSamples / monoData.length;
      const maxSilenceDuration = maxSilenceSamples / sampleRate;

      let rmsDynamicRange = 0;
      if (rmsWindows.length > 0) {
        rmsDynamicRange = Math.max(...rmsWindows) - Math.min(...rmsWindows);
      }

      const rms = Math.sqrt(sumSquares / monoData.length);
      const peakDB = peak > 0 ? 20 * Math.log10(peak) : -Infinity;
      const rmsDB = rms > 0 ? 20 * Math.log10(rms) : -Infinity;
      const dcOffset = dcOffsetSum / monoData.length;

      setStats({
        format: selectedFile.type || selectedFile.name.split('.').pop()?.toUpperCase() || 'Unknown',
        sampleRate,
        numChannels,
        duration,
        peakDB,
        rmsDB,
        clipCount,
        dcOffset,
        noiseLevelDB,
        snrDB,
        silenceRatio,
        maxSilenceDuration,
        rmsDynamicRange
      });
    } catch (err) {
      console.error(err);
      setError("Failed to analyze audio. The file might be corrupted or unsupported.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const audioBufferToWav = (buffer: AudioBuffer, startSample: number, endSample: number): File => {
    const sampleRate = buffer.sampleRate;
    const length = endSample - startSample;
    const wavBuffer = new ArrayBuffer(44 + length * 2);
    const view = new DataView(wavBuffer);
    const writeStr = (offset: number, str: string) => {
      for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
    };
    writeStr(0, 'RIFF');
    view.setUint32(4, 36 + length * 2, true);
    writeStr(8, 'WAVE');
    writeStr(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true); // mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeStr(36, 'data');
    view.setUint32(40, length * 2, true);
    // mix down to mono
    const numChannels = buffer.numberOfChannels;
    let offset = 44;
    for (let i = startSample; i < endSample; i++) {
      let sum = 0;
      for (let c = 0; c < numChannels; c++) sum += buffer.getChannelData(c)[i];
      const s = Math.max(-1, Math.min(1, sum / numChannels));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
      offset += 2;
    }
    return new File([wavBuffer], 'chunk.wav', { type: 'audio/wav' });
  };

  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}m ${s}s`;
  };

  const downloadCSV = () => {
    if (!parsedAiData?.ai_analysis?.transcript_by_turn) return;
    const turns = parsedAiData.ai_analysis.transcript_by_turn;
    const taskId = file?.name.split('.')[0] || 'audio';

    const headers = [
      'Task ID', 'Turn No.', 'Speaker', 'Start Time', 'End Time',
      'Original Utterance',
      'Emotion', 'Intent', 'Speaking Rate',
      'Disfluency: None', 'Disfluency: Filler', 'Disfluency: False Start',
      'Disfluency: Self-repair', 'Disfluency: Repetition', 'Disfluency: Long Pause',
      'Turn-taking Event', 'Emphasis', 'Annotator Notes'
    ];

    const escapeCell = (val: string) => `"${String(val ?? '').replace(/"/g, '""')}"`;
    // Convert snake_case / lowercase values to Title Case for display
    const toTitle = (val: string) =>
      (val ?? '').replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());

    const rows = turns.map((turn: any) => {
      const ann = turn.annotations || {};
      const disfluencies: string[] = Array.isArray(ann.disfluency) ? ann.disfluency : [];
      const emphasis: string[] = Array.isArray(ann.emphasis) ? ann.emphasis : [];

      return [
        escapeCell(taskId),
        escapeCell(String(turn.turn_id ?? '')),
        escapeCell(turn.speaker ?? ''),
        escapeCell(turn.start_time ?? ''),
        escapeCell(turn.end_time ?? ''),
        escapeCell(turn.text ?? ''),
        escapeCell(toTitle(ann.emotion ?? '')),
        escapeCell(toTitle(ann.intent ?? '')),
        escapeCell(toTitle(ann.speaking_rate ?? '')),
        escapeCell(disfluencies.includes('none') ? 'Yes' : 'No'),
        escapeCell(disfluencies.includes('filler') ? 'Yes' : 'No'),
        escapeCell(disfluencies.includes('false_start') ? 'Yes' : 'No'),
        escapeCell(disfluencies.includes('self_repair') ? 'Yes' : 'No'),
        escapeCell(disfluencies.includes('repetition') ? 'Yes' : 'No'),
        escapeCell(disfluencies.includes('long_pause') ? 'Yes' : 'No'),
        escapeCell(toTitle(ann.turn_taking ?? '')),
        escapeCell(emphasis.join(', ')),
        escapeCell(''),
      ].join(',');
    });

    const csvContent = [headers.map(escapeCell).join(','), ...rows].join('\n');
    const blob = new Blob(['\uFEFF' + csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${taskId}_transcript_annotation.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const runAiAnalysis = async () => {
    if (!file) return;
    setIsAiAnalyzing(true);
    setAiError(null);
    setAiResult(null); 
    setAiJsonResult(null);

    try {
      // Fetch runtime API key from backend
      const configRes = await fetch('/api/config');
      if (!configRes.ok) throw new Error("Failed to fetch API configuration");
      const config = await configRes.json();
      
      const apiKey = config.GEMINI_API_KEY;
      const transcriptApiKey = config.TRANSCRIPT_API_KEY || apiKey; // Fallback to primary if not set
      const jsonApiKey = config.JSON_API_KEY || apiKey; // Fallback to primary if not set
      if (!apiKey && !jsonApiKey) {
        throw new Error("API key is missing from environment variables.");
      }

      const withRetry = async <T,>(fn: () => Promise<T>, maxRetries = 4, baseDelay = 2000): Promise<T> => {
        let attempt = 0;
        while (attempt < maxRetries) {
          try {
            return await fn();
          } catch (error: any) {
            const isRetryable = error?.status === 503 || error?.message?.includes('503') || error?.message?.includes('high demand') || error?.status === 429;
            if (isRetryable && attempt < maxRetries - 1) {
              attempt++;
              const delay = baseDelay * Math.pow(2, attempt - 1) + Math.random() * 1000;
              console.warn(`API error (503/429). Retrying attempt ${attempt} in ${Math.round(delay)}ms...`);
              await new Promise(resolve => setTimeout(resolve, delay));
            } else {
              throw error;
            }
          }
        }
        throw new Error("Max retries reached");
      };

      const aiAnalysis = new GoogleGenAI({ apiKey: jsonApiKey });
      const aiTranscript = new GoogleGenAI({ apiKey: transcriptApiKey });

      // 1. Build audio chunks (60s each) from stored AudioBuffer
      const CHUNK_SECONDS = 60;
      const audioBuffer = audioBufferRef.current;
      const chunkFiles: { file: File; offsetSec: number }[] = [];

      if (audioBuffer && audioBuffer.duration > CHUNK_SECONDS) {
        const sr = audioBuffer.sampleRate;
        const chunkSamples = sr * CHUNK_SECONDS;
        for (let start = 0; start < audioBuffer.length; start += chunkSamples) {
          const end = Math.min(start + chunkSamples, audioBuffer.length);
          chunkFiles.push({ file: audioBufferToWav(audioBuffer, start, end), offsetSec: start / sr });
        }
      } else {
        // Audio is short enough — treat the original file as a single chunk
        chunkFiles.push({ file: file, offsetSec: 0 });
      }

      // 2. Upload full file (for analysis) + all chunks (for transcript) in parallel
      const [uploadResultAnalysis, ...chunkUploadResults] = await Promise.all([
        withRetry(() => aiAnalysis.files.upload({
          file: file,
          config: { mimeType: file.type || 'audio/mp3', displayName: `analysis-${file.name}` }
        })),
        ...chunkFiles.map(({ file: chunkFile }, idx) =>
          withRetry(() => aiTranscript.files.upload({
            file: chunkFile,
            config: { mimeType: 'audio/wav', displayName: `chunk-${idx}-${file.name}` }
          }))
        )
      ]);

      const fileUriAnalysis = uploadResultAnalysis.uri;

      // 2. Generate JSON Analysis directly
      setIsGeneratingJson(true);
      try {
        // Content safety — full file
        const analysisPromise = withRetry(() => aiAnalysis.models.generateContent({
          model: 'gemini-3.1-pro-preview',
          contents: [
            {
              fileData: {
                fileUri: fileUriAnalysis,
                mimeType: file.type || 'audio/mp3'
              }
            },
            `Analyze the audio and provide a JSON response with ONLY the following structure. Do NOT include transcripts. ALL SCORES MUST BE NUMBERS (e.g. 0.5), NOT STRINGS:
{
  "content_safety_scores": {
    "toxicity_detected": "Number 0.0 to 1.0 (Reject >= 0.3)",
    "sexual_content_detected": "Number 0.0 to 1.0 (Reject >= 0.2)",
    "violent_intent": "Number 0.0 to 1.0 (Reject >= 0.5)",
    "political_campaigning": "Number 0.0 to 1.0 (Reject >= 0.5)",
    "discriminatory_content": "Number 0.0 to 1.0 (Reject >= 0.3)",
    "pii_leakage": "Number (Count of real PII entities like phone, Aadhaar, PAN. Reject if > 0)"
  },
  "conversation_quality_metrics": {
    "is_scripted": "Number 0.0 to 1.0 (Reject >= 0.85)",
    "real_world_impersonation": "Number 0.0 to 1.0 (Reject >= 0.5)",
    "excessive_code_switching": "Number 0.0 to 1.0 (Reject > 0.3)",
    "topic_coherence_fail": "Number 0.0 to 1.0 (Reject >= 0.75)",
    "native_speaker_fail": "Number 0.0 to 1.0 (Reject >= 0.5)",
    "task_alignment_fail": "Number 0.0 to 1.0 (Reject >= 0.5)",
    "emotion_sentiment_mismatch": "Number 0.0 to 1.0 (Reject >= 0.5)"
  },
  "voice_quality_metrics": {
    "unnatural_pauses": "Number 0.0 to 1.0 (Reject >= 0.5)",
    "robotic_tone": "Number 0.0 to 1.0 (Reject >= 0.5)",
    "audio_glitches": "Number 0.0 to 1.0 (Reject >= 0.5)"
  }
}`
          ],
          config: { responseMimeType: "application/json" }
        }));

        // Transcript — one request per chunk, all in parallel
        const transcriptPromises = chunkUploadResults.map((uploadResult, idx) => {
          const offsetSec = chunkFiles[idx].offsetSec;
          return withRetry(() => aiTranscript.models.generateContent({
            model: 'gemini-3.1-pro-preview',
            contents: [
              {
                fileData: {
                  fileUri: uploadResult.uri,
                  mimeType: 'audio/wav'
                }
              },
              `You are an expert verbatim audio transcriber with deep knowledge of Indian languages and scripts.

**STEP 1 — Identify the language FIRST before transcribing anything.**
Listen carefully to the phonology and vocabulary of the audio and determine the exact language spoken. Indian languages sound similar but use completely different scripts:
- Kannada sounds like Telugu but uses a DIFFERENT script: ಕನ್ನಡ (Kannada) ≠ తెలుగు (Telugu)
- Hindi uses Devanagari (देवनागरी) — NOT the same as Gujarati or Marathi even though they share the same script family
- Tamil (தமிழ்) is completely distinct from all other South Indian languages

Common Indian languages and their ONLY correct scripts:
  • Kannada   → ಕನ್ನಡ ಲಿಪಿ   (e.g. ಹೋದಾ, ತಗೊಂಡು, ಏನು, ಮಾಡು, ಬಾ, ಅವರು)
  • Telugu    → తెలుగు లిపి  (e.g. వెళ్ళారు, తీసుకున్నారు, ఏమి, చేయండి)
  • Hindi     → देवनागरी     (e.g. गया, लिया, क्या, करो, आप)
  • Tamil     → தமிழ் எழுத்து (e.g. போனாங்க, எடுத்தாங்க, என்ன, செய்)
  • Malayalam → മലയാളം       (e.g. പോയി, എടുത്തു, എന്ത്, ചെയ്യ്)
  • Marathi   → मराठी (देवनागरी) (e.g. गेलो, घेतलं, काय, कर)

**NEVER mix up scripts. If the audio is Kannada, every native word MUST be in ಕನ್ನಡ script — not Telugu, not Hindi, not Roman.**

This audio starts at ${Math.floor(offsetSec / 60)}m ${Math.floor(offsetSec % 60)}s in the original recording. Generate a transcript in JSON format exactly like this:
{
  "detected_language": "The exact language spoken (e.g. Kannada, Telugu, Hindi, Tamil, mixed Kannada-English). Be precise — do not guess.",
  "speakers": ["Speaker 1", "Speaker 2"],
  "transcript_by_turn": [
    {
      "speaker": "Speaker 1",
      "start_time": "MM:SS",
      "end_time": "MM:SS",
      "text": "Verbatim text with all inline tags as specified in the rules below."
    }
  ]
}
IMPORTANT: All timestamps must be absolute (offset from start of full recording, not this chunk). This chunk starts at ${Math.floor(offsetSec / 60).toString().padStart(2, '0')}:${Math.floor(offsetSec % 60).toString().padStart(2, '0')}.

**TRANSCRIPTION RULES — follow ALL rules without exception:**

**1. Strict Verbatim**
- Transcribe every word exactly as spoken.
- Include every filler (um, uh, hmm, etc.) exactly as heard.
- Tag structural disfluencies inline at the exact point they occur: use [false_start] before a false start and [repetition] before a repeated word/phrase.
- Do NOT fix grammar, reorder words, or paraphrase.

**2. Acoustic Events**
- Every audible non-speech event must be tagged inline where it occurs.
- Use ONLY these tags: [noise], [laughter], [cough], [music], [silence], [unintelligible].

**3. Timestamps & Turns**
- Each speaker turn must have start_time and end_time in MM:SS format.
- For overlapping speech, write each speaker as a separate turn entry with their actual start/end timestamps — do NOT add notes like "(overlapping)".

**4. Entity Tagging (BIO Format)**
- Tag ALL named entities inline using BIO format — even inside non-English turns.
- Base format: [B-TYPE] FirstWord [I-TYPE] NextWord [/TYPE]
- Allowed entity types: PERSON, ORG, LOCATION, DATE, TIME, MONEY, QUANTITY, MEDICAL_TERM, LEGAL_TERM, PRODUCT

**CRITICAL — [LANG:EN] wrapper rule:**
ANY entity whose words are in English (Latin script) and which appears inside a non-English (e.g. Hindi, Tamil, Telugu) sentence MUST be wrapped with [LANG:EN]...[/LANG:EN] around the entire BIO tag. This rule applies to EVERY entity type without exception.

Correct examples (English entity inside Hindi sentence):
- PERSON:   हाँ [LANG:EN][B-PERSON] Prince [/PERSON][/LANG:EN], कैसे हो?
- ORG:      मैं [LANG:EN][B-ORG] Google [/ORG][/LANG:EN] पे काम करता हूँ।
- ORG multi-word: [LANG:EN][B-ORG] Google [I-ORG] Drive [/ORG][/LANG:EN] use करो।
- PRODUCT:  उसने [LANG:EN][B-PRODUCT] iPhone [/PRODUCT][/LANG:EN] खरीदा।
- LOCATION: वो [LANG:EN][B-LOCATION] Mumbai [/LOCATION][/LANG:EN] गया।
- DATE:     meeting [LANG:EN][B-DATE] 5th [I-DATE] March [/DATE][/LANG:EN] को है।

Wrong (entity in Hindi with no wrapper — NEVER do this):
- हाँ [B-PERSON] Prince [/PERSON], कैसे हो?  ← WRONG

**Entity vs. Loanword — the exact boundary:**
Ask yourself: Is this word a SPECIFIC named thing (a brand, a person's name, a place name, a named product)?
→ YES → it is a named entity → apply BIO tag + [LANG:EN] wrapper.
→ NO  → it is a generic English loanword → keep in English script only, NO BIO tag.

Named entity examples (MUST tag):
- "Google", "Amazon", "iCloud", "WhatsApp" → ORG or PRODUCT
- "Prince", "Riya", "John" → PERSON
- "Mumbai", "Delhi", "London" → LOCATION
- "iPhone", "Samsung Galaxy", "Google Photos" → PRODUCT
- "Google Drive", "Dropbox", "OneDrive" → PRODUCT (named cloud services)

Generic loanword examples (do NOT tag — Rule 6 applies):
- "photos", "photo", "video", "file" → generic nouns
- "cloud storage", "storage", "internet" → generic tech concepts, not a specific named service
- "upload", "download", "save", "backup" → verbs/actions
- "phone", "mobile", "laptop", "computer" → generic device nouns
- "companies", "apps", "service", "plan" → generic category nouns
- "plus", "like", "basically", "actually" → discourse fillers/connectors

**5. Orthography & Numbers**
- Apply full and correct punctuation (commas, periods, question marks, etc.).
- Capitalize the first letter of every sentence and all proper nouns.
- Numbers below 100: write in words (e.g., "thirty-two").
- Numbers 100 and above: write in digits (e.g., "150").
- Dates: always write in digits (e.g., "04/04/2026").

**6. Multilingual & Script Rules**
- Native Script: ALWAYS transcribe non-English speech in the correct native Unicode script. NEVER romanize or transliterate native speech.
- Script must match the detected language exactly:
  • Kannada audio   → ALL native words in ಕನ್ನಡ script   (NEVER use Telugu తెలుగు or Hindi देवनागरी for Kannada)
  • Telugu audio    → ALL native words in తెలుగు script   (NEVER use Kannada ಕನ್ನಡ)
  • Hindi audio     → ALL native words in देवनागरी        (NEVER use ಕನ್ನಡ or తెలుగు)
  • Tamil audio     → ALL native words in தமிழ் script
  • Malayalam audio → ALL native words in മലയാളം script
- Kannada vs Telugu confusion warning: These two languages sound similar but are completely different. Kannada example: "ಅದು ಎಲ್ಲಿ ಇದೆ?" — Telugu equivalent would be: "అది ఎక్కడ ఉంది?" — they are NOT interchangeable.
- Everyday English loanwords (e.g., "oh", "bag", "phone", "ok", "yes") spoken inside a native-language sentence: keep those specific words in English script, not transliterated.
- Beeps / Sensitive Info: If a beep masks PII (name, DOB, phone number), write [beep]. Never guess the hidden content.
- Cut-off sentences: If a speaker is interrupted mid-sentence, end the text with a dash —.`
            ],
            config: { responseMimeType: "application/json" }
          }));
        });

        const [analysisRes, ...chunkTranscriptResults] = await Promise.all([analysisPromise, ...transcriptPromises]);

        let aiDataObj = {};
        let transcriptDataObj: any = {};

        try {
          const rawAnalysis = analysisRes.text || "{}";
          // Strategy 1: strip markdown fences then parse
          let cleanAnalysis = rawAnalysis.replace(/```json/gi, '').replace(/```/g, '').trim();
          try {
            aiDataObj = JSON.parse(cleanAnalysis);
          } catch {
            // Strategy 2: extract outermost {...} block in case model added surrounding text
            const firstBrace = rawAnalysis.indexOf('{');
            const lastBrace = rawAnalysis.lastIndexOf('}');
            if (firstBrace !== -1 && lastBrace > firstBrace) {
              cleanAnalysis = rawAnalysis.slice(firstBrace, lastBrace + 1);
              aiDataObj = JSON.parse(cleanAnalysis);
            } else {
              throw new Error("No JSON object found in analysis response");
            }
          }
          console.log("Analysis parsed OK, keys:", Object.keys(aiDataObj));
        } catch (e) {
          console.error("Analysis JSON parse error:", e, "\nRaw response:", analysisRes.text?.slice(0, 500));
        }


        // Merge transcript chunks into a single transcript
        const allTurns: any[] = [];
        let detectedLanguage = '';
        const speakerSet = new Set<string>();

        for (const chunkRes of chunkTranscriptResults) {
          try {
            const raw = chunkRes.text || "{}";
            let clean = raw.replace(/```json/gi, '').replace(/```/g, '').trim();
            let parsed: any;
            try {
              parsed = JSON.parse(clean);
            } catch {
              const fb = raw.indexOf('{'), lb = raw.lastIndexOf('}');
              if (fb !== -1 && lb > fb) parsed = JSON.parse(raw.slice(fb, lb + 1));
              else throw new Error("No JSON in chunk response");
            }
            if (parsed.detected_language && !detectedLanguage) detectedLanguage = parsed.detected_language;
            (parsed.speakers || []).forEach((s: string) => speakerSet.add(s));
            (parsed.transcript_by_turn || []).forEach((t: any) => allTurns.push(t));
          } catch (e) { console.error("Chunk transcript parse error:", e); }
        }

        // 3. Annotation step — enrich each turn with conversation labels
        // Always assign turn_id so turns have it even if annotation step fails
        let annotatedTurns = allTurns.map((turn, idx) => ({ turn_id: idx + 1, ...turn }));
        if (allTurns.length > 0) {
          try {
            // Strip BIO/LANG tags before sending to annotation — raw brackets in mixed-script
            // text cause the annotation model to emit malformed JSON.
            const stripTags = (text: string) =>
              text
                .replace(/\[LANG:[A-Z]+\]/g, '')
                .replace(/\[\/LANG:[A-Z]+\]/g, '')
                .replace(/\[B-[A-Z_]+\]/g, '')
                .replace(/\[I-[A-Z_]+\]/g, '')
                .replace(/\[\/[A-Z_]+\]/g, '')
                .replace(/\s+/g, ' ')
                .trim();

            const transcriptText = allTurns.map((t, idx) =>
              `[turn_id:${idx + 1}] [${t.start_time} --> ${t.end_time}] ${t.speaker}: ${stripTags(t.text)}`
            ).join('\n');

            const annotationRes = await withRetry(() => aiAnalysis.models.generateContent({
              model: 'gemini-3.1-pro-preview',
              contents: [`You are a conversation analyst. Analyze the transcript below and return ONLY a valid JSON object — no markdown, no commentary — with this exact structure:

{
  "conversation_analysis": {
    "emotional_arc": "<Stable / consistent tone throughout|Escalation (calm → tense)|De-escalation (tense → calm)|Escalation → resolution|Fluctuating / no clear pattern>",
    "dominant_dynamic": "<Collaborative|Negotiation|Conflict|Information exchange|Instructional (one speaker leading)|Social / casual>",
    "turn_taking_pattern": "<Balanced|Speaker A dominant|Speaker B dominant|Frequent overlaps / interruptions|Long monologues with minimal back-and-forth>",
    "background_noise_level": "<Clean / silent|Low (AC hum, distant sounds)|Moderate (office, street, keyboard)|High (crowd, traffic, music)>"
  },
  "turns": [
    {
      "turn_id": <integer starting at 1>,
      "speaker": "<speaker label>",
      "timestamp_start": "<start time>",
      "timestamp_end": "<end time>",
      "duration_seconds": <end minus start in seconds, 2 decimal places>,
      "gap_from_previous_seconds": <gap since previous turn ended, 0.0 for first turn>,
      "text": "<original text verbatim>",
      "annotations": {
        "emotion": "<neutral|frustrated|anxious|confident|skeptical|amused|excited|confused|resigned|sarcastic>",
        "disfluency": ["<none|filler|false_start|self_repair|repetition|long_pause>"],
        "speaking_rate": "<slow|normal|fast>",
        "turn_taking": "<normal_transition|latch|overlap|interruption|long_gap>",
        "emphasis": ["<stressed words>"],
        "intent": "<statement|question|proposal|agreement|disagreement|correction|request|elaboration|backchannel>"
      }
    }
  ]
}

RULES:

turn_taking classification — use EXACTLY one of: latch | normal_transition | overlap | interruption | long_gap
  • latch          gap_from_previous_seconds 0 – 0.25s  → response begins almost instantly, no gap or overlap, turns feel seamlessly glued; listener anticipated the ending; common in fast-paced or familiar exchanges
  • normal_transition  gap 0.25 – 1.25s              → short comfortable pause, most natural pacing, enough time to process and respond, no awkwardness
  • overlap        gap_from_previous_seconds < 0 AND the previous speaker completes (or nearly completes) their thought → next speaker starts while previous is still talking, no intent to cut off, both voices briefly audible; signals agreement / enthusiasm / anticipation
  • interruption   gap_from_previous_seconds < 0 AND the previous speaker is forced to stop mid-point → deliberate cut-off, original speaker does not finish; signals dominance, urgency, disagreement, or frustration
  • long_gap       gap_from_previous_seconds ≥ 1.25s  → prolonged uncomfortable silence; signals hesitation, confusion, careful thinking, emotional processing, or disengagement

- First turn: turn_taking = "normal_transition", gap_from_previous_seconds = 0.0
- For overlap vs interruption: if the prior speaker's text ends with "—" (cut off) → interruption; otherwise → overlap
- WPM = (word_count / duration_seconds) × 60: <120 → slow, 120–170 → normal, >170 → fast
- disfluency: pick ALL that apply; use ["none"] if clean
- emphasis: list words that carry notable stress or communicative weight in the turn. Since only text is available (no audio), infer stress from textual and semantic signals:
    • Intensifiers / degree adverbs (e.g. "very", "extremely", "খুবই", "অনেক", "বিলকুল", "बिल्कुल")
    • Words in a correction turn (intent = "correction") — the corrected word is typically stressed
    • Repeated or re-stated words within the same turn (repetition for emphasis)
    • Words in exclamatory or emotionally heightened turns (emotion = "frustrated" / "excited" / "confident") that carry the core emotional load
    • Contrastive words explicitly juxtaposing two ideas (e.g. "না … হ্যাঁ", "not X but Y")
    • Code-switched words inserted for rhetorical punch (e.g. English words like "seriously", "actually" inside a Bengali sentence)
    • Use [] only if none of the above signals are present
- intent: statement = sharing info; question = asking for info; proposal = suggesting idea; agreement = expressing agreement; disagreement = pushing back; correction = fixing an error; request = asking for action; elaboration = adding detail to prior turn; backchannel = acknowledgment only (mm-hmm, right, okay)
- background_noise_level: infer from audio quality cues in the transcript (e.g. [noise], [beep], interruptions) and spoken context

TRANSCRIPT:
${transcriptText}`],
            }));

            const rawAnnotation = annotationRes.text || "{}";
            const cleanAnnotation = rawAnnotation.replace(/```json/gi, '').replace(/```/g, '').trim();
            const annotationData: { conversation_analysis?: any; turns?: any[] } = JSON.parse(cleanAnnotation);
            const annotationTurns: any[] = annotationData.turns || [];
            console.log("Annotation data received:", annotationTurns.length, "turns");
            annotatedTurns = allTurns.map((turn, idx) => {
              const annotated = annotationTurns[idx] || annotationTurns.find((a: any) => a.turn_id === idx + 1);
              return annotated ? { turn_id: idx + 1, ...turn, annotations: annotated.annotations } : { turn_id: idx + 1, ...turn };
            });
            if (annotationData.conversation_analysis) {
              (transcriptDataObj as any).conversation_analysis = annotationData.conversation_analysis;
            }
          } catch (e) {
            console.error("Annotation step failed:", e);
          }
        }

        transcriptDataObj = {
          ...transcriptDataObj,
          detected_language: detectedLanguage,
          speakers: Array.from(speakerSet),
          transcript_by_turn: annotatedTurns
        };

        const aiData = { ...aiDataObj, ...transcriptDataObj };

        const programmaticData = {
          duration_seconds: Number(stats?.duration.toFixed(2) || 0),
          snr_db: Number(stats?.snrDB.toFixed(2) || 0),
          clipping_detected: stats?.clipCount ? stats.clipCount > 0 : false,
          background_noise_dbfs: Number(stats?.noiseLevelDB.toFixed(2) || 0),
          dc_offset: Number(stats?.dcOffset.toFixed(6) || 0),
          silence_ratio_percent: Number(((stats?.silenceRatio || 0) * 100).toFixed(2)),
          prosody_per_turn: { pitch_hz: null, intensity_db: null },
          diarization_rttm: null
        };

        const masterJson = {
          audio_metrics: programmaticData,
          ai_analysis: aiData
        };

        setAiJsonResult(JSON.stringify(masterJson, null, 2));
      } catch (err) {
        console.error("JSON Generation Error:", err);
        throw err;
      } finally {
        setIsGeneratingJson(false);
      }

    } catch (err: any) {
      console.error(err);
      let errorMessage = err.message || "Failed to run AI analysis. Please check your API key and try again.";
      try {
        const errorJson = JSON.parse(errorMessage);
        if (errorJson.error && errorJson.error.message) {
          errorMessage = errorJson.error.message;
        }
      } catch(e) {
        // If it's not valid JSON, just keep the original error message
      }
      setAiError(errorMessage);
    } finally {
      setIsAiAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-neutral-50 text-neutral-900 p-8 font-sans">
      <div className="max-w-5xl mx-auto space-y-8">
        <header className="space-y-2">
          <h1 className="text-3xl font-bold tracking-tight flex items-center gap-3">
            <Activity className="w-8 h-8 text-blue-600" />
            Audio Quality-Control Dashboard
          </h1>
          <p className="text-neutral-500 text-lg">
            Upload an audio file to run deterministic quality checks and AI-powered semantic analysis.
          </p>
        </header>

        <section className="bg-white p-8 rounded-2xl shadow-sm border border-neutral-200">
          <div className="flex flex-col items-center justify-center border-2 border-dashed border-neutral-300 rounded-xl p-12 bg-neutral-50 hover:bg-neutral-100 transition-colors cursor-pointer relative">
            <input 
              type="file" 
              accept="audio/wav,audio/mpeg,audio/flac,audio/*" 
              onChange={handleFileUpload}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            />
            <Upload className="w-12 h-12 text-neutral-400 mb-4" />
            <p className="text-lg font-medium text-neutral-700">Click or drag to upload audio</p>
            <p className="text-neutral-500 text-sm mt-1">Supports WAV, MP3, FLAC</p>
          </div>

          {error && (
            <div className="mt-6 p-4 bg-red-50 text-red-700 rounded-lg flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 shrink-0 mt-0.5" />
              <p>{error}</p>
            </div>
          )}

          {isAnalyzing && (
            <div className="mt-6 flex items-center justify-center gap-3 text-neutral-500">
              <div className="animate-spin rounded-full h-5 w-5 border-2 border-blue-600 border-t-transparent"></div>
              Analyzing audio waveform...
            </div>
          )}

          {file && !isAnalyzing && !error && (
            <div className="mt-6">
              <audio ref={audioRef} src={URL.createObjectURL(file)} controls className="w-full" />
            </div>
          )}
        </section>

        {stats && (
          <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <section>
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <FileAudio className="w-5 h-5 text-neutral-500" />
                Metadata & Format
              </h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <MetricCard label="Format" value={stats.format} />
                <MetricCard label="Sample Rate" value={`${stats.sampleRate} Hz`} />
                <MetricCard label="Channels" value={stats.numChannels === 1 ? 'Mono' : stats.numChannels === 2 ? 'Stereo' : stats.numChannels.toString()} />
                <MetricCard label="Duration" value={formatTime(stats.duration)} />
              </div>
            </section>

            <section>
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Activity className="w-5 h-5 text-neutral-500" />
                Functional Quality Checks
              </h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <StatusCard 
                  label="Clipping (0 dBFS)" 
                  isWarning={stats.clipCount > 0}
                  value={stats.clipCount > 0 ? `Detected (${stats.clipCount} samples)` : 'No Clipping'}
                />
                
                <StatusCard 
                  label="Background Noise" 
                  isWarning={stats.noiseLevelDB > noiseThreshold}
                  value={`${stats.noiseLevelDB.toFixed(1)} dBFS`}
                />

                <StatusCard 
                  label="Signal-to-Noise Ratio" 
                  isWarning={stats.snrDB < snrThreshold}
                  value={`${stats.snrDB.toFixed(1)} dB`}
                />

                <StatusCard 
                  label="Silence Ratio" 
                  isWarning={(stats.silenceRatio * 100) > maxSilenceRatio}
                  value={`${(stats.silenceRatio * 100).toFixed(1)}%`}
                />

                <StatusCard 
                  label="Max Silence Duration" 
                  isWarning={false}
                  value={`${stats.maxSilenceDuration.toFixed(2)}s`}
                />

                <StatusCard 
                  label="RMS Dynamic Range" 
                  isWarning={false}
                  value={`${stats.rmsDynamicRange.toFixed(1)} dB`}
                />

                <StatusCard 
                  label="Duration Check" 
                  isWarning={stats.duration < minDuration || stats.duration > maxDuration}
                  value={stats.duration < minDuration ? 'Too Short' : stats.duration > maxDuration ? 'Too Long' : 'Passed'}
                />

                <StatusCard 
                  label="Speaker Count" 
                  isWarning={parsedAiData?.ai_analysis?.speakers ? parsedAiData.ai_analysis.speakers.length !== expectedSpeakers : false}
                  value={parsedAiData?.ai_analysis?.speakers ? `${parsedAiData.ai_analysis.speakers.length} (Expected ${expectedSpeakers})` : 'Pending AI'}
                />
              </div>
              
              {Math.abs(stats.dcOffset) > 0.01 && (
                <div className="mt-4 p-4 bg-amber-50 text-amber-800 rounded-lg flex items-start gap-3 border border-amber-200">
                  <AlertTriangle className="w-5 h-5 shrink-0 mt-0.5 text-amber-600" />
                  <div>
                    <p className="font-medium">Significant DC Offset Detected</p>
                    <p className="text-sm mt-1">The waveform is not centered at zero (Offset: {stats.dcOffset.toFixed(4)}). This can reduce headroom and cause clicks during editing.</p>
                  </div>
                </div>
              )}

              {parsedAiData?.ai_analysis?.content_safety_scores ? (
                <div className="mt-8">
                  <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-neutral-700">
                    <Shield className="w-5 h-5 text-neutral-500" />
                    Content Safety
                  </h3>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    <ScoreCard label="Toxicity Detected" value={parsedAiData.ai_analysis.content_safety_scores.toxicity_detected || 0} threshold=">= 0.3" isWarning={(parsedAiData.ai_analysis.content_safety_scores.toxicity_detected || 0) >= 0.3} />
                    <ScoreCard label="Sexual Content Detected" value={parsedAiData.ai_analysis.content_safety_scores.sexual_content_detected || 0} threshold=">= 0.2" isWarning={(parsedAiData.ai_analysis.content_safety_scores.sexual_content_detected || 0) >= 0.2} />
                    <ScoreCard label="Violent Intent" value={parsedAiData.ai_analysis.content_safety_scores.violent_intent || 0} threshold=">= 0.5" isWarning={(parsedAiData.ai_analysis.content_safety_scores.violent_intent || 0) >= 0.5} />
                    <ScoreCard label="Political Campaigning" value={parsedAiData.ai_analysis.content_safety_scores.political_campaigning || 0} threshold=">= 0.5" isWarning={(parsedAiData.ai_analysis.content_safety_scores.political_campaigning || 0) >= 0.5} />
                    <ScoreCard label="Discriminatory Content" value={parsedAiData.ai_analysis.content_safety_scores.discriminatory_content || 0} threshold=">= 0.3" isWarning={(parsedAiData.ai_analysis.content_safety_scores.discriminatory_content || 0) >= 0.3} />
                    <ScoreCard label="PII Leakage" value={parsedAiData.ai_analysis.content_safety_scores.pii_leakage || 0} threshold="> 0" isWarning={(parsedAiData.ai_analysis.content_safety_scores.pii_leakage || 0) > 0} isInteger={true} />
                  </div>
                </div>
              ) : (
                <div className="mt-8">
                  <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-neutral-700">
                    <Shield className="w-5 h-5 text-neutral-500" />
                    Content Safety
                  </h3>
                  <div className={`border border-dashed rounded-xl p-6 flex flex-col items-center justify-center ${aiJsonResult ? 'bg-amber-50 border-amber-300 text-amber-700' : 'bg-neutral-50 border-neutral-200 text-neutral-500'}`}>
                    {aiJsonResult ? <AlertTriangle className="w-8 h-8 mb-2 text-amber-500" /> : <BrainCircuit className="w-8 h-8 mb-2 text-neutral-400" />}
                    <p>{aiJsonResult ? 'Analysis response could not be parsed. Re-run AI Analysis.' : 'Click "Run AI Analysis" below to run AI-powered safety checks.'}</p>
                  </div>
                </div>
              )}

              {parsedAiData?.ai_analysis?.conversation_quality_metrics ? (
                <div className="mt-8">
                  <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-neutral-700">
                    <MessageSquare className="w-5 h-5 text-neutral-500" />
                    Conversation Quality
                  </h3>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    <ScoreCard label="Is Scripted" value={parsedAiData.ai_analysis.conversation_quality_metrics.is_scripted || 0} threshold=">= 0.85" isWarning={(parsedAiData.ai_analysis.conversation_quality_metrics.is_scripted || 0) >= 0.85} />
                    <ScoreCard label="Real World Impersonation" value={parsedAiData.ai_analysis.conversation_quality_metrics.real_world_impersonation || 0} threshold=">= 0.5" isWarning={(parsedAiData.ai_analysis.conversation_quality_metrics.real_world_impersonation || 0) >= 0.5} />
                    <ScoreCard label="Excessive Code Switching" value={parsedAiData.ai_analysis.conversation_quality_metrics.excessive_code_switching || 0} threshold="> 0.3" isWarning={(parsedAiData.ai_analysis.conversation_quality_metrics.excessive_code_switching || 0) > 0.3} />
                    <ScoreCard label="Topic Coherence Fail" value={parsedAiData.ai_analysis.conversation_quality_metrics.topic_coherence_fail || 0} threshold=">= 0.75" isWarning={(parsedAiData.ai_analysis.conversation_quality_metrics.topic_coherence_fail || 0) >= 0.75} />
                    <ScoreCard label="Native Speaker Fail" value={parsedAiData.ai_analysis.conversation_quality_metrics.native_speaker_fail || 0} threshold=">= 0.5" isWarning={(parsedAiData.ai_analysis.conversation_quality_metrics.native_speaker_fail || 0) >= 0.5} />
                    <ScoreCard label="Task Alignment Fail" value={parsedAiData.ai_analysis.conversation_quality_metrics.task_alignment_fail || 0} threshold=">= 0.5" isWarning={(parsedAiData.ai_analysis.conversation_quality_metrics.task_alignment_fail || 0) >= 0.5} />
                    <ScoreCard label="Emotion Sentiment Mismatch" value={parsedAiData.ai_analysis.conversation_quality_metrics.emotion_sentiment_mismatch || 0} threshold=">= 0.5" isWarning={(parsedAiData.ai_analysis.conversation_quality_metrics.emotion_sentiment_mismatch || 0) >= 0.5} />
                  </div>
                </div>
              ) : (
                <div className="mt-8">
                  <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-neutral-700">
                    <MessageSquare className="w-5 h-5 text-neutral-500" />
                    Conversation Quality
                  </h3>
                  <div className={`border border-dashed rounded-xl p-6 flex flex-col items-center justify-center ${aiJsonResult ? 'bg-amber-50 border-amber-300 text-amber-700' : 'bg-neutral-50 border-neutral-200 text-neutral-500'}`}>
                    {aiJsonResult ? <AlertTriangle className="w-8 h-8 mb-2 text-amber-500" /> : <BrainCircuit className="w-8 h-8 mb-2 text-neutral-400" />}
                    <p>{aiJsonResult ? 'Analysis response could not be parsed. Re-run AI Analysis.' : 'Click "Run AI Analysis" below to run AI-powered quality checks.'}</p>
                  </div>
                </div>
              )}

              {parsedAiData?.ai_analysis?.voice_quality_metrics ? (
                <div className="mt-8">
                  <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-neutral-700">
                    <Activity className="w-5 h-5 text-neutral-500" />
                    Voice Quality
                  </h3>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    <ScoreCard label="Unnatural Pauses" value={parsedAiData.ai_analysis.voice_quality_metrics.unnatural_pauses || 0} threshold=">= 0.5" isWarning={(parsedAiData.ai_analysis.voice_quality_metrics.unnatural_pauses || 0) >= 0.5} />
                    <ScoreCard label="Robotic Tone" value={parsedAiData.ai_analysis.voice_quality_metrics.robotic_tone || 0} threshold=">= 0.5" isWarning={(parsedAiData.ai_analysis.voice_quality_metrics.robotic_tone || 0) >= 0.5} />
                    <ScoreCard label="Audio Glitches" value={parsedAiData.ai_analysis.voice_quality_metrics.audio_glitches || 0} threshold=">= 0.5" isWarning={(parsedAiData.ai_analysis.voice_quality_metrics.audio_glitches || 0) >= 0.5} />
                  </div>
                </div>
              ) : null}

              {parsedAiData?.ai_analysis?.transcript_by_turn ? (
                <div className="mt-8">
                  <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-neutral-700">
                    <MessageSquare className="w-5 h-5 text-neutral-500" />
                    Transcript
                    {parsedAiData.ai_analysis.detected_language && (
                      <span className="ml-2 text-xs font-medium bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full">
                        {parsedAiData.ai_analysis.detected_language}
                      </span>
                    )}
                  </h3>
                  <div className="bg-white p-6 rounded-xl border border-neutral-200 shadow-sm space-y-4">
                    {parsedAiData.ai_analysis.transcript_by_turn.map((turn: any, idx: number) => (
                      <div key={idx} className="flex flex-col sm:flex-row gap-2 sm:gap-4">
                        <div className="sm:w-32 shrink-0">
                          <span className="font-semibold text-blue-700">{turn.speaker}</span>
                          <div className="text-xs text-neutral-500">{turn.start_time} - {turn.end_time}</div>
                        </div>
                        <div className="text-neutral-800">
                          {turn.text}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}

              {aiJsonResult && (
                <div className="mt-8">
                  <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-neutral-700">
                    <Settings2 className="w-5 h-5 text-neutral-500" />
                    Raw JSON Output
                  </h3>
                  <div className="bg-neutral-900 text-neutral-100 p-6 rounded-xl overflow-x-auto text-sm font-mono shadow-inner">
                    <pre>{aiJsonResult}</pre>
                  </div>
                </div>
              )}
            </section>

            <section className="bg-blue-50/50 p-8 rounded-2xl border border-blue-100">
              <div className="flex items-start justify-between mb-6">
                <div className="flex-1 pr-4">
                  <h2 className="text-xl font-semibold flex items-center gap-2 text-blue-900">
                    <BrainCircuit className="w-6 h-6 text-blue-600" />
                    AI Semantic Analysis & Transcription
                  </h2>
                  <p className="text-blue-700/80 mt-1">Use Gemini 3.1 Pro to detect speakers, generate a highly accurate transcript, and run content safety checks.</p>
                </div>
                <button 
                  onClick={runAiAnalysis}
                  disabled={isAiAnalyzing || isGeneratingJson}
                  className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2.5 rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 whitespace-nowrap"
                >
                  {(isAiAnalyzing || isGeneratingJson) ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-2 border-white/30 border-t-white"></div>
                      Analyzing...
                    </>
                  ) : 'Run AI Analysis'}
                </button>
              </div>

              {aiError && (
                <div className="p-4 bg-red-50 text-red-700 rounded-lg flex items-start gap-3 border border-red-100 mb-6">
                  <AlertTriangle className="w-5 h-5 shrink-0 mt-0.5" />
                  <p>{aiError}</p>
                </div>
              )}

              {aiJsonResult && (
                <div className="flex flex-col sm:flex-row gap-4 mt-6">
                  <button
                    onClick={() => {
                      const blob = new Blob([aiJsonResult], { type: 'application/json' });
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement('a');
                      a.href = url;
                      a.download = `${file?.name.split('.')[0] || 'audio'}_analysis.json`;
                      document.body.appendChild(a);
                      a.click();
                      document.body.removeChild(a);
                      URL.revokeObjectURL(url);
                    }}
                    className="flex items-center justify-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-xl font-medium hover:bg-blue-700 transition-colors shadow-sm w-full sm:w-auto"
                  >
                    <Download className="w-5 h-5" />
                    Download JSON Analysis
                  </button>
                  {parsedAiData?.ai_analysis?.transcript_by_turn && (
                    <button
                      onClick={downloadCSV}
                      className="flex items-center justify-center gap-2 px-6 py-3 bg-emerald-600 text-white rounded-xl font-medium hover:bg-emerald-700 transition-colors shadow-sm w-full sm:w-auto"
                    >
                      <Download className="w-5 h-5" />
                      Download CSV Annotation
                    </button>
                  )}
                </div>
              )}
            </section>
          </div>
        )}
      </div>
    </div>
  );
}



function MetricCard({ label, value }: { label: string, value: string | number }) {
  return (
    <div className="bg-white p-5 rounded-xl border border-neutral-200 shadow-sm">
      <p className="text-sm font-medium text-neutral-500 mb-1">{label}</p>
      <p className="text-2xl font-semibold text-neutral-900">{value}</p>
    </div>
  );
}

function StatusCard({ label, isWarning, value }: { label: string, isWarning: boolean, value: string | number }) {
  return (
    <div className={`bg-white p-5 rounded-xl border shadow-sm ${isWarning ? 'border-red-200 bg-red-50' : 'border-neutral-200'}`}>
      <div className="flex items-start justify-between mb-2">
        <p className={`text-sm font-medium ${isWarning ? 'text-red-700' : 'text-neutral-500'}`}>{label}</p>
        {isWarning ? <AlertTriangle className="w-5 h-5 text-red-500" /> : <CheckCircle className="w-5 h-5 text-emerald-500" />}
      </div>
      <p className={`text-xl font-semibold ${isWarning ? 'text-red-900' : 'text-neutral-900'}`}>{value}</p>
    </div>
  );
}

function ScoreCard({ label, value, threshold, isWarning, isInteger = false }: { label: string, value: any, threshold: string, isWarning: boolean, isInteger?: boolean }) {
  const numValue = Number(value) || 0;
  return (
    <div className={`p-4 rounded-xl border shadow-sm ${isWarning ? 'bg-red-50 border-red-200' : 'bg-emerald-50 border-emerald-200'}`}>
      <p className={`text-xs font-semibold uppercase tracking-wider mb-1 ${isWarning ? 'text-red-700' : 'text-emerald-700'}`}>
        {label}
      </p>
      <div className="flex items-center gap-2 mb-1">
        {isWarning ? <AlertTriangle className="w-5 h-5 text-red-600" /> : <CheckCircle className="w-5 h-5 text-emerald-600" />}
        <p className={`text-2xl font-bold ${isWarning ? 'text-red-900' : 'text-emerald-900'}`}>
          {isInteger ? Math.round(numValue) : numValue.toFixed(2)}
        </p>
      </div>
      <p className={`text-xs font-medium ${isWarning ? 'text-red-600' : 'text-emerald-600'}`}>
        {isWarning ? `Fail (${threshold})` : 'Pass'}
      </p>
    </div>
  );
}
